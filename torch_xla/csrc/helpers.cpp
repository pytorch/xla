#include "torch_xla/csrc/helpers.h"

#include <limits>

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

xla::XlaOp ConvertBinaryOpResult(xla::XlaOp op1, xla::XlaOp op2,
                                 xla::XlaOp result) {
  xla::PrimitiveType type1 = XlaHelpers::TypeOfXlaOp(op1);
  xla::PrimitiveType type2 = XlaHelpers::TypeOfXlaOp(op2);
  xla::PrimitiveType result_type = XlaHelpers::TypeOfXlaOp(result);
  if (type1 == type2 && type1 != result_type) {
    return ConvertTo(result, result_type, type1, /*device=*/nullptr);
  }
  return result;
}

}  // namespace

xla::PrecisionConfig::Precision XlaHelpers::s_mat_mul_precision =
    xla::PrecisionConfig::DEFAULT;

xla::PrecisionConfig XlaHelpers::BuildPrecisionConfig(
    const xla::PrecisionConfig::Precision conv_precision) {
  xla::PrecisionConfig precision_config;
  // Dot and convolution take two operators.
  precision_config.mutable_operand_precision()->Resize(
      /*new_size=*/2, conv_precision);
  return precision_config;
}

xla::XlaComputation CreateComputation(
    const std::string& name, xla::PrimitiveType type,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& op) {
  xla::XlaBuilder builder(name);
  xla::XlaOp x =
      xla::Parameter(&builder, 0, xla::ShapeUtil::MakeShape(type, {}), "x");
  xla::XlaOp y =
      xla::Parameter(&builder, 1, xla::ShapeUtil::MakeShape(type, {}), "y");
  return ConsumeValue(builder.Build(op(x, y)));
}

xla::XlaOp XlaHelpers::CreateReturnValue(
    xla::XlaBuilder* builder, const std::vector<xla::XlaOp>& outputs) {
  if (outputs.size() > 1) {
    return xla::Tuple(builder, outputs);
  } else if (!outputs.empty()) {
    return xla::GetTupleElement(xla::Tuple(builder, {outputs[0]}), 0);
  } else {
    return xla::Tuple(builder, {});
  }
}

std::vector<xla::int64> XlaHelpers::DropDimensions(
    absl::Span<const xla::int64> sizes,
    absl::Span<const xla::int64> drop_dims) {
  std::vector<xla::int64> new_dims;
  size_t drop_index = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (drop_index < drop_dims.size() && i == drop_dims[drop_index]) {
      ++drop_index;
    } else {
      new_dims.push_back(sizes[i]);
    }
  }
  XLA_CHECK_EQ(drop_index, drop_dims.size());
  return new_dims;
}

xla::int64 XlaHelpers::GetCanonicalDimensionIndex(xla::int64 dim,
                                                  xla::int64 rank) {
  xla::int64 min_shape_dim = -rank;
  xla::int64 max_shape_dim = rank - 1;
  XLA_CHECK(min_shape_dim <= dim && dim <= max_shape_dim)
      << "Value out of range (expected to be in range of [" << min_shape_dim
      << ", " << max_shape_dim << "], but got " << dim << ")";
  xla::int64 dim_index = dim < 0 ? rank + dim : dim;
  XLA_CHECK_GE(dim_index, 0);
  XLA_CHECK_LT(dim_index, rank);
  return dim_index;
}

std::vector<xla::int64> XlaHelpers::GetCanonicalDimensionIndices(
    absl::Span<const xla::int64> dimensions, xla::int64 rank) {
  std::vector<xla::int64> canonical_dim_indices;
  for (xla::int64 dim : dimensions) {
    canonical_dim_indices.push_back(GetCanonicalDimensionIndex(dim, rank));
  }
  return canonical_dim_indices;
}

xla::int64 XlaHelpers::GetCanonicalPosition(
    absl::Span<const xla::int64> dimensions, xla::int64 dim, xla::int64 pos) {
  dim = GetCanonicalDimensionIndex(dim, dimensions.size());
  if (pos < 0) {
    pos = GetCanonicalDimensionIndex(pos, dimensions[dim]);
  } else {
    pos = std::min<xla::int64>(pos, dimensions[dim]);
  }
  return pos;
}

xla::int64 XlaHelpers::GetDynamicDimension(const xla::Shape& shape) {
  xla::int64 dynamic_dimension = -1;
  for (xla::int64 i = 0; i < shape.rank(); ++i) {
    if (shape.is_dynamic_dimension(i)) {
      XLA_CHECK(dynamic_dimension < 0)
          << "Only one dynamic dimension is supported: " << i << " and "
          << dynamic_dimension << " in " << shape;
      dynamic_dimension = i;
    }
  }
  return dynamic_dimension;
}

XlaHelpers::DynamicSize XlaHelpers::GetDimensionsSize(
    absl::Span<const xla::XlaOp> inputs,
    absl::Span<const xla::int64> dimensions) {
  XLA_CHECK(!inputs.empty());
  xla::PrimitiveType size_type = GetShapeDimensionType(/*device=*/nullptr);
  xla::XlaOp size;
  xla::int64 size_scalar = 1;
  for (auto& input : inputs) {
    const xla::Shape& shape = ShapeOfXlaOp(input);
    for (auto dim : dimensions) {
      if (size_scalar >= 0) {
        if (!shape.is_dynamic_dimension(dim)) {
          size_scalar *= shape.dimensions(dim);
          continue;
        } else {
          if (size_scalar != 1) {
            size = ScalarValue(size_scalar, size_type, input.builder());
          }
          size_scalar = -1;
        }
      }
      if (size.valid()) {
        size = size * xla::GetDimensionSize(input, dim);
      } else {
        size = xla::GetDimensionSize(input, dim);
      }
    }
  }
  absl::optional<xla::int64> scalar_size;
  if (size_scalar >= 0) {
    scalar_size = size_scalar;
  }
  if (!size.valid()) {
    size = ScalarValue(size_scalar, size_type, inputs[0].builder());
  }
  return {size, scalar_size};
}

XlaHelpers::MinMax XlaHelpers::MinMaxValues(xla::PrimitiveType type) {
  switch (type) {
    case xla::PrimitiveType::S8:
      return {std::numeric_limits<xla::int8>::lowest(),
              std::numeric_limits<xla::int8>::max()};
    case xla::PrimitiveType::U8:
      return {std::numeric_limits<xla::uint8>::lowest(),
              std::numeric_limits<xla::uint8>::max()};
    case xla::PrimitiveType::S16:
      return {std::numeric_limits<xla::int16>::lowest(),
              std::numeric_limits<xla::int16>::max()};
    case xla::PrimitiveType::U16:
      return {std::numeric_limits<xla::uint16>::lowest(),
              std::numeric_limits<xla::uint16>::max()};
    case xla::PrimitiveType::S32:
      return {static_cast<int64_t>(std::numeric_limits<xla::int32>::lowest()),
              static_cast<int64_t>(std::numeric_limits<xla::int32>::max())};
    case xla::PrimitiveType::U32:
      return {static_cast<int64_t>(std::numeric_limits<xla::uint32>::lowest()),
              static_cast<int64_t>(std::numeric_limits<xla::uint32>::max())};
    case xla::PrimitiveType::S64:
      return {static_cast<int64_t>(std::numeric_limits<xla::int64>::lowest()),
              static_cast<int64_t>(std::numeric_limits<xla::int64>::max())};
    case xla::PrimitiveType::U64:
      return {static_cast<int64_t>(std::numeric_limits<xla::uint64>::lowest()),
              static_cast<int64_t>(std::numeric_limits<xla::uint64>::max())};
    case xla::PrimitiveType::BF16:
    case xla::PrimitiveType::F32:
      return {std::numeric_limits<float>::lowest(),
              std::numeric_limits<float>::max()};
    case xla::PrimitiveType::F64:
      return {std::numeric_limits<double>::lowest(),
              std::numeric_limits<double>::max()};
    case xla::PrimitiveType::PRED:
      return {0, 1};
    default:
      XLA_ERROR() << "Unsupported XLA type " << type;
  }
}

xla::PaddingConfig XlaHelpers::MakeXlaPaddingConfigFromNdPadding(
    absl::Span<const xla::int64> padding) {
  XLA_CHECK_EQ(padding.size() % 2, 0)
      << "Padding specification must have even length";
  XLA_CHECK(!padding.empty()) << "Padding specification cannot be empty";
  xla::PaddingConfig padding_config;
  for (int i = 0; i < padding.size(); i += 2) {
    xla::PaddingConfig::PaddingConfigDimension* dims =
        padding_config.add_dimensions();
    dims->set_edge_padding_low(padding[padding.size() - i - 2]);
    dims->set_edge_padding_high(padding[padding.size() - i - 1]);
  }
  return padding_config;
}

xla::XlaComputation XlaHelpers::CreateAddComputation(xla::PrimitiveType type) {
  return CreateComputation(
      "AddComputation", type, [&](xla::XlaOp x, xla::XlaOp y) {
        return type == xla::PrimitiveType::PRED ? xla::Or(x, y)
                                                : xla::Add(x, y);
      });
}

xla::XlaComputation XlaHelpers::CreateMulComputation(xla::PrimitiveType type) {
  return CreateComputation(
      "MulComputation", type,
      [&](xla::XlaOp x, xla::XlaOp y) { return xla::Mul(x, y); });
}

xla::XlaComputation XlaHelpers::CreateMaxComputation(xla::PrimitiveType type) {
  return CreateComputation(
      "MaxComputation", type,
      [&](xla::XlaOp x, xla::XlaOp y) { return xla::Max(x, y); });
}

xla::XlaComputation XlaHelpers::CreateMinComputation(xla::PrimitiveType type) {
  return CreateComputation(
      "MinComputation", type,
      [&](xla::XlaOp x, xla::XlaOp y) { return xla::Min(x, y); });
}

xla::XlaComputation XlaHelpers::CreateAndComputation(xla::PrimitiveType type) {
  return CreateComputation(
      "AndComputation", type,
      [&](xla::XlaOp x, xla::XlaOp y) { return xla::And(x, y); });
}

xla::XlaComputation XlaHelpers::CreateOrComputation(xla::PrimitiveType type) {
  return CreateComputation(
      "OrComputation", type,
      [&](xla::XlaOp x, xla::XlaOp y) { return xla::Or(x, y); });
}

const xla::Shape& XlaHelpers::ShapeOfXlaOp(xla::XlaOp op) {
  const xla::Shape* shape = ConsumeValue(op.builder()->GetShapePtr(op));
  return *shape;
}

std::vector<xla::int64> XlaHelpers::SizesOfXlaOp(xla::XlaOp op) {
  const xla::Shape& op_shape = ShapeOfXlaOp(op);
  return std::vector<xla::int64>(op_shape.dimensions().begin(),
                                 op_shape.dimensions().end());
}

xla::PrimitiveType XlaHelpers::TypeOfXlaOp(xla::XlaOp op) {
  return ShapeOfXlaOp(op).element_type();
}

xla::XlaOp XlaHelpers::ReshapeToRank(xla::XlaOp input, xla::int64 expected_rank,
                                     xla::int64 offset) {
  const xla::Shape& shape = ShapeOfXlaOp(input);
  XLA_CHECK_LE(offset + shape.rank(), expected_rank);
  if (shape.rank() == expected_rank) {
    return input;
  }
  std::vector<xla::int64> dimensions(expected_rank - offset - shape.rank(), 1);
  dimensions.insert(dimensions.end(), shape.dimensions().begin(),
                    shape.dimensions().end());
  dimensions.insert(dimensions.end(), offset, 1);
  return xla::Reshape(input, dimensions);
}

absl::optional<XlaHelpers::DynamicReshapeInfo>
XlaHelpers::GetDynamicReshapeInfo(const xla::Shape& input_shape,
                                  absl::Span<const xla::int64> output_sizes) {
  xla::int64 input_dynamic_dimension = GetDynamicDimension(input_shape);
  if (input_dynamic_dimension < 0) {
    return absl::nullopt;
  }
  DynamicReshapeInfo info;
  info.output_shape =
      xla::ShapeUtil::MakeShape(input_shape.element_type(), output_sizes);
  if (info.output_shape.rank() > 0) {
    xla::int64 size_at_dyndim = 1;
    for (xla::int64 i = 0; i <= input_dynamic_dimension; ++i) {
      size_at_dyndim *= input_shape.dimensions(i);
    }
    xla::int64 dynamic_dimension = -1;
    xla::int64 out_size = 1;
    for (xla::int64 i = 0; i < output_sizes.size(); ++i) {
      XLA_CHECK_LE(out_size, size_at_dyndim / input_shape.dimensions(
                                                  input_dynamic_dimension))
          << "Unable to map dynamic dimension of shape " << input_shape
          << " to output sizes (" << absl::StrJoin(output_sizes, ", ") << ")";
      out_size *= output_sizes[i];
      if (out_size >= size_at_dyndim) {
        dynamic_dimension = i;
        break;
      }
    }
    XLA_CHECK(dynamic_dimension >= 0)
        << "Unable to map dynamic dimension of shape " << input_shape
        << " to output sizes (" << absl::StrJoin(output_sizes, ", ") << ")";
    info.dynamic_dimension = dynamic_dimension;
    info.output_shape.set_dynamic_dimension(info.dynamic_dimension, true);
  }
  return std::move(info);
}

xla::Shape XlaHelpers::GetDynamicReshape(
    const xla::Shape& input_shape, absl::Span<const xla::int64> output_sizes) {
  auto info = GetDynamicReshapeInfo(input_shape, output_sizes);
  if (info) {
    return info->output_shape;
  }
  return xla::ShapeUtil::MakeShape(input_shape.element_type(), output_sizes);
}

xla::XlaOp XlaHelpers::DynamicReshape(
    xla::XlaOp input, absl::Span<const xla::int64> output_sizes) {
  const xla::Shape& input_shape = ShapeOfXlaOp(input);
  if (output_sizes == input_shape.dimensions()) {
    return input;
  }
  auto info = GetDynamicReshapeInfo(input_shape, output_sizes);
  if (info) {
    return xla::ReshapeWithInferredDimension(input, output_sizes,
                                             info->dynamic_dimension);
  }
  return xla::Reshape(input, output_sizes);
}

xla::XlaOp XlaHelpers::DynamicReshapeAs(xla::XlaOp input,
                                        const xla::Shape& shape) {
  const xla::Shape& input_shape = ShapeOfXlaOp(input);
  xla::int64 dynamic_dimension = GetDynamicDimension(shape);
  if (dynamic_dimension >= 0) {
    return xla::ReshapeWithInferredDimension(input, shape.dimensions(),
                                             dynamic_dimension);
  }
  return shape.dimensions() == input_shape.dimensions()
             ? input
             : xla::Reshape(input, shape.dimensions());
}

bool XlaHelpers::SameStaticDimensions(const xla::Shape& shape1,
                                      const xla::Shape& shape2) {
  return shape1.is_static() && shape2.is_static() &&
         shape1.dimensions() == shape2.dimensions();
}

xla::XlaOp XlaHelpers::Flatten(xla::XlaOp input, xla::Shape* input_shape) {
  xla::util::MaybePtr<xla::Shape> input_shape_tmp(input_shape);
  *input_shape_tmp = ShapeOfXlaOp(input);
  if (input_shape_tmp->rank() == 1) {
    return input;
  }
  xla::int64 input_elements = xla::ShapeUtil::ElementsIn(*input_shape_tmp);
  return DynamicReshape(input, {input_elements});
}

std::vector<xla::int64> XlaHelpers::MakeTransposePermutation(xla::int64 dim0,
                                                             xla::int64 dim1,
                                                             xla::int64 rank) {
  xla::int64 canonical_dim0 = GetCanonicalDimensionIndex(dim0, rank);
  xla::int64 canonical_dim1 = GetCanonicalDimensionIndex(dim1, rank);
  auto permute_dims = xla::util::Iota<xla::int64>(rank);
  std::swap(permute_dims[canonical_dim0], permute_dims[canonical_dim1]);
  return permute_dims;
}

xla::XlaOp XlaHelpers::LinearInterpolation(xla::XlaOp value0, xla::XlaOp value1,
                                           double alpha) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(value0);
  xla::XlaOp one = xla::One(value0.builder(), shape.element_type());
  xla::XlaOp alpha_value =
      ScalarValue(alpha, shape.element_type(), value0.builder());
  return value0 * alpha_value + value1 * (one - alpha_value);
}

xla::PrimitiveType XlaHelpers::PromoteType(xla::PrimitiveType type1,
                                           xla::PrimitiveType type2) {
  if (type1 == type2) {
    return type1;
  }
  xla::int64 size1 = xla::ShapeUtil::ByteSizeOfPrimitiveType(type1);
  xla::int64 size2 = xla::ShapeUtil::ByteSizeOfPrimitiveType(type2);
  if (xla::primitive_util::IsFloatingPointType(type1)) {
    return (!xla::primitive_util::IsFloatingPointType(type2) || size1 >= size2)
               ? type1
               : type2;
  }
  if (xla::primitive_util::IsFloatingPointType(type2) || size2 > size1) {
    return type2;
  }
  if (xla::primitive_util::IsIntegralType(type1) &&
      xla::primitive_util::IsIntegralType(type2)) {
    return size1 >= size2 ? type1 : type2;
  }
  if (type1 == xla::PrimitiveType::PRED) {
    return type2;
  }
  if (type2 == xla::PrimitiveType::PRED) {
    return type1;
  }
  // If nothing matches the above logic, first operand wins.
  return type1;
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteValues(xla::XlaOp op1,
                                                            xla::XlaOp op2) {
  xla::PrimitiveType type1 = TypeOfXlaOp(op1);
  xla::PrimitiveType type2 = TypeOfXlaOp(op2);
  xla::PrimitiveType result_type = PromoteType(type1, type2);
  if (type1 != result_type) {
    op1 = ConvertTo(op1, type1, result_type, /*device=*/nullptr);
  }
  if (type2 != result_type) {
    op2 = ConvertTo(op2, type2, result_type, /*device=*/nullptr);
  }
  return std::pair<xla::XlaOp, xla::XlaOp>(op1, op2);
}

std::tuple<xla::XlaOp, xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteValues(
    xla::XlaOp op1, xla::XlaOp op2, xla::XlaOp op3) {
  xla::PrimitiveType type1 = TypeOfXlaOp(op1);
  xla::PrimitiveType type2 = TypeOfXlaOp(op2);
  xla::PrimitiveType type3 = TypeOfXlaOp(op3);
  xla::PrimitiveType result_type =
      PromoteType(PromoteType(type1, type2), type3);
  if (type1 != result_type) {
    op1 = ConvertTo(op1, type1, result_type, /*device=*/nullptr);
  }
  if (type2 != result_type) {
    op2 = ConvertTo(op2, type2, result_type, /*device=*/nullptr);
  }
  if (type3 != result_type) {
    op3 = ConvertTo(op3, type3, result_type, /*device=*/nullptr);
  }
  return std::tuple<xla::XlaOp, xla::XlaOp, xla::XlaOp>(op1, op2, op3);
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteSecondValue(
    xla::XlaOp op1, xla::XlaOp op2) {
  xla::PrimitiveType type1 = TypeOfXlaOp(op1);
  xla::PrimitiveType type2 = TypeOfXlaOp(op2);
  return type1 == type2
             ? std::pair<xla::XlaOp, xla::XlaOp>(op1, op2)
             : std::pair<xla::XlaOp, xla::XlaOp>(
                   op1, ConvertTo(op2, type2, type1, /*device=*/nullptr));
}

std::vector<xla::int64> XlaHelpers::GetPromotedShape(
    absl::Span<const xla::int64> shape1_dims,
    absl::Span<const xla::int64> shape2_dims) {
  std::vector<xla::int64> dimensions;
  // If the rank of a shape is bigger than then other, fill up the first
  // dimensions with the ones of the bigger.
  // Example:
  //   shape1 = [9, 7, 6, 5, 2]
  //   shape2 =       [6, 1, 2]
  // Insert [9, 7] into the dimensions vector.
  if (shape1_dims.size() > shape2_dims.size()) {
    dimensions.insert(
        dimensions.end(), shape1_dims.begin(),
        shape1_dims.begin() + (shape1_dims.size() - shape2_dims.size()));
  } else if (shape2_dims.size() > shape1_dims.size()) {
    dimensions.insert(
        dimensions.end(), shape2_dims.begin(),
        shape2_dims.begin() + (shape2_dims.size() - shape1_dims.size()));
  }
  // For the common dimensions, they must match, or one of them be 1.
  size_t min_size = std::min(shape1_dims.size(), shape2_dims.size());
  for (xla::int64 i = 0; i < min_size; ++i) {
    xla::int64 dim1 = shape1_dims[shape1_dims.size() - min_size + i];
    xla::int64 dim2 = shape2_dims[shape2_dims.size() - min_size + i];
    XLA_CHECK(dim1 == dim2 || dim1 == 1 || dim2 == 1)
        << "(" << absl::StrJoin(shape1_dims, ", ") << ") and ("
        << absl::StrJoin(shape1_dims, ", ") << ")";
    if (dim1 == 0 || dim2 == 0) {
      dimensions.push_back(0);
    } else {
      dimensions.push_back(std::max<xla::int64>(dim1, dim2));
    }
  }
  return dimensions;
}

xla::Shape XlaHelpers::GetPromotedShape(const xla::Shape& shape1,
                                        const xla::Shape& shape2) {
  return xla::ShapeUtil::MakeShape(
      shape1.element_type(),
      GetPromotedShape(shape1.dimensions(), shape2.dimensions()));
}

xla::Shape XlaHelpers::GetPromotedBinaryOpShape(const xla::Shape& shape1,
                                                const xla::Shape& shape2) {
  return xla::ShapeUtil::MakeShape(
      PromoteType(shape1.element_type(), shape2.element_type()),
      GetPromotedShape(shape1.dimensions(), shape2.dimensions()));
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteShapes(xla::XlaOp op1,
                                                            xla::XlaOp op2) {
  const xla::Shape& shape1 = ShapeOfXlaOp(op1);
  const xla::Shape& shape2 = ShapeOfXlaOp(op2);
  if (xla::ShapeUtil::Compatible(shape1, shape2)) {
    // Fast path shortcut if the shapes already matches in dimensions.
    return std::pair<xla::XlaOp, xla::XlaOp>(op1, op2);
  }
  XLA_CHECK(xla::ShapeUtil::SameElementType(shape1, shape2))
      << shape1 << " and " << shape2;

  xla::Shape shape = GetPromotedShape(shape1, shape2);
  return std::pair<xla::XlaOp, xla::XlaOp>(
      ImplicitBroadcast(op1, shape1, shape),
      ImplicitBroadcast(op2, shape2, shape));
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::Promote(xla::XlaOp op1,
                                                      xla::XlaOp op2) {
  std::pair<xla::XlaOp, xla::XlaOp> vops = PromoteValues(op1, op2);
  return PromoteShapes(vops.first, vops.second);
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteSecond(xla::XlaOp op1,
                                                            xla::XlaOp op2) {
  std::pair<xla::XlaOp, xla::XlaOp> vops = PromoteSecondValue(op1, op2);
  return PromoteShapes(vops.first, vops.second);
}

xla::XlaOp XlaHelpers::ImplicitBroadcast(xla::XlaOp op,
                                         const xla::Shape& op_shape,
                                         const xla::Shape& shape) {
  const auto& op_shape_dims = op_shape.dimensions();
  const auto& shape_dims = shape.dimensions();
  XLA_CHECK_GE(shape_dims.size(), op_shape_dims.size())
      << shape << " vs " << op_shape;
  xla::int64 size_delta = shape_dims.size() - op_shape_dims.size();
  xla::XlaOp new_op = op;
  if (!std::equal(op_shape_dims.begin(), op_shape_dims.end(),
                  shape_dims.begin() + size_delta)) {
    // If the base N dimensions do not match, broadcast the original op.
    // Example:
    //   op_shape =       [3, 1, 5]
    //   shape    = [6, 8, 3, 4, 5]
    // After this operation we will have:
    //   op_shape =       [3, 4, 5]
    std::vector<xla::int64> common_shape_dims(shape_dims.begin() + size_delta,
                                              shape_dims.end());
    std::vector<xla::int64> broadcast_dimensions(op_shape_dims.size());
    std::iota(broadcast_dimensions.begin(), broadcast_dimensions.end(), 0);
    new_op =
        xla::BroadcastInDim(new_op, common_shape_dims, broadcast_dimensions);
  }
  if (size_delta > 0) {
    // Add the major dimensions if necessary:
    // Example:
    //   op_shape =       [3, 4, 5]
    //   shape    = [6, 8, 3, 4, 5]
    // After this operation we will have (added [6, 8]):
    //   op_shape = [6, 8, 3, 4, 5]
    std::vector<xla::int64> broadcast_sizes(shape_dims.begin(),
                                            shape_dims.begin() + size_delta);
    new_op = xla::Broadcast(new_op, broadcast_sizes);
  }
  return new_op;
}

xla::XlaOp XlaHelpers::PromotedBinaryOp(
    xla::XlaOp op1, xla::XlaOp op2,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& bin_op) {
  xla::XlaOp numeric_op1 = ConvertToNumeric(op1);
  xla::XlaOp numeric_op2 = ConvertToNumeric(op2);
  std::pair<xla::XlaOp, xla::XlaOp> vops = Promote(numeric_op1, numeric_op2);
  xla::XlaOp result = bin_op(vops.first, vops.second);
  return ConvertBinaryOpResult(op1, op2, result);
}

}  // namespace torch_xla
