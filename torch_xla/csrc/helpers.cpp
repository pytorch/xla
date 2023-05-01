#include "torch_xla/csrc/helpers.h"

#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/util.h>

#include <limits>

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/sys_util.h"
#include "third_party/xla_client/tf_logging.h"
#include "third_party/xla_client/util.h"
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

}  // namespace

xla::PrecisionConfig::Precision XlaHelpers::s_mat_mul_precision =
    xla::PrecisionConfig::DEFAULT;

xla::PrecisionConfig XlaHelpers::BuildPrecisionConfig(
    xla::PrecisionConfig::Precision conv_precision, int num_arguments) {
  xla::PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(num_arguments,
                                                       conv_precision);
  return precision_config;
}

xla::XlaOp XlaHelpers::BroadcastDimensions(xla::XlaOp input,
                                           absl::Span<const int64_t> dimensions,
                                           absl::Span<const int64_t> sizes) {
  XLA_CHECK_EQ(dimensions.size(), sizes.size());
  std::vector<int64_t> bcast_sizes = SizesOfXlaOp(input);
  for (size_t i = 0; i < dimensions.size(); ++i) {
    bcast_sizes.at(dimensions[i]) = sizes[i];
  }
  return xla::BroadcastInDim(input, bcast_sizes,
                             GetAllDimensions(bcast_sizes.size()));
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

int64_t XlaHelpers::GetDynamicDimension(const xla::Shape& shape) {
  int64_t dynamic_dimension = -1;
  for (int64_t i = 0; i < shape.rank(); ++i) {
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
    absl::Span<const xla::XlaOp> inputs, absl::Span<const int64_t> dimensions) {
  XLA_CHECK(!inputs.empty());
  xla::PrimitiveType size_type = GetShapeDimensionType(/*device=*/nullptr);
  xla::XlaOp size;
  int64_t size_scalar = 1;
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
  absl::optional<int64_t> scalar_size;
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
      return {std::numeric_limits<int8_t>::lowest(),
              std::numeric_limits<int8_t>::max()};
    case xla::PrimitiveType::U8:
      return {std::numeric_limits<uint8_t>::lowest(),
              std::numeric_limits<uint8_t>::max()};
    case xla::PrimitiveType::S16:
      return {std::numeric_limits<int16_t>::lowest(),
              std::numeric_limits<int16_t>::max()};
    case xla::PrimitiveType::U16:
      return {std::numeric_limits<uint16_t>::lowest(),
              std::numeric_limits<uint16_t>::max()};
    case xla::PrimitiveType::S32:
      return {static_cast<int64_t>(std::numeric_limits<int32_t>::lowest()),
              static_cast<int64_t>(std::numeric_limits<int32_t>::max())};
    case xla::PrimitiveType::U32:
      return {static_cast<int64_t>(std::numeric_limits<uint32_t>::lowest()),
              static_cast<int64_t>(std::numeric_limits<uint32_t>::max())};
    case xla::PrimitiveType::S64:
      return {static_cast<int64_t>(std::numeric_limits<int64_t>::lowest()),
              static_cast<int64_t>(std::numeric_limits<int64_t>::max())};
    case xla::PrimitiveType::U64:
      return {static_cast<int64_t>(std::numeric_limits<uint64_t>::lowest()),
              static_cast<int64_t>(std::numeric_limits<uint64_t>::max())};
    case xla::PrimitiveType::F16:
      return {static_cast<float>(std::numeric_limits<xla::half>::lowest()),
              static_cast<float>(std::numeric_limits<xla::half>::max())};
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
    absl::Span<const int64_t> padding) {
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

std::vector<int64_t> XlaHelpers::SizesOfXlaOp(xla::XlaOp op) {
  const xla::Shape& op_shape = ShapeOfXlaOp(op);
  return std::vector<int64_t>(op_shape.dimensions().begin(),
                              op_shape.dimensions().end());
}

xla::PrimitiveType XlaHelpers::TypeOfXlaOp(xla::XlaOp op) {
  return ShapeOfXlaOp(op).element_type();
}

xla::XlaOp XlaHelpers::ReshapeToRank(xla::XlaOp input, int64_t expected_rank,
                                     int64_t offset) {
  const xla::Shape& shape = ShapeOfXlaOp(input);
  XLA_CHECK_LE(offset + shape.rank(), expected_rank);
  if (shape.rank() == expected_rank) {
    return input;
  }
  std::vector<int64_t> dimensions(expected_rank - offset - shape.rank(), 1);
  dimensions.insert(dimensions.end(), shape.dimensions().begin(),
                    shape.dimensions().end());
  dimensions.insert(dimensions.end(), offset, 1);
  return xla::Reshape(input, dimensions);
}

absl::optional<XlaHelpers::DynamicReshapeInfo>
XlaHelpers::GetDynamicReshapeInfo(const xla::Shape& input_shape,
                                  absl::Span<const int64_t> output_sizes) {
  int64_t input_dyndim_idx = GetDynamicDimension(input_shape);
  if (input_dyndim_idx < 0) {
    return absl::nullopt;
  }
  DynamicReshapeInfo info;
  info.output_shape =
      xla::ShapeUtil::MakeShape(input_shape.element_type(), output_sizes);
  if (info.output_shape.rank() > 0) {
    int64_t size_prod_until_dyndim = 1;
    for (int64_t i = 0; i <= input_dyndim_idx; ++i) {
      size_prod_until_dyndim *= input_shape.dimensions(i);
    }
    int64_t dynamic_dimension = -1;
    int64_t out_size = 1;
    for (int64_t i = 0; i < output_sizes.size(); ++i) {
      XLA_CHECK_LE(out_size, size_prod_until_dyndim /
                                 input_shape.dimensions(input_dyndim_idx))
          << "Unable to map dynamic dimension of shape " << input_shape
          << " to output sizes (" << absl::StrJoin(output_sizes, ", ") << ")";
      out_size *= output_sizes[i];
      if (out_size >= size_prod_until_dyndim) {
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
    const xla::Shape& input_shape, absl::Span<const int64_t> output_sizes) {
  auto info = GetDynamicReshapeInfo(input_shape, output_sizes);
  if (info) {
    return info->output_shape;
  }
  return xla::ShapeUtil::MakeShape(input_shape.element_type(), output_sizes);
}

xla::XlaOp XlaHelpers::DynamicReshape(xla::XlaOp input,
                                      absl::Span<const int64_t> output_sizes) {
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
  int64_t dynamic_dimension = GetDynamicDimension(shape);
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
  int64_t input_elements = xla::ShapeUtil::ElementsIn(*input_shape_tmp);
  return DynamicReshape(input, {input_elements});
}

xla::XlaOp XlaHelpers::FlattenDimRange(xla::XlaOp input, int64_t start,
                                       int64_t range, xla::Shape* input_shape) {
  xla::util::MaybePtr<xla::Shape> input_shape_tmp(input_shape);
  *input_shape_tmp = ShapeOfXlaOp(input);

  std::vector<int64_t> sizes;
  int64_t flat_size = -1;
  for (int64_t dim = 0; dim < input_shape_tmp->rank(); ++dim) {
    if (dim < start || dim >= start + range) {
      if (flat_size >= 0) {
        sizes.push_back(flat_size);
        flat_size = -1;
      }
      sizes.push_back(input_shape_tmp->dimensions(dim));
    } else {
      flat_size =
          (flat_size < 0 ? 1 : flat_size) * input_shape_tmp->dimensions(dim);
    }
  }
  if (flat_size >= 0) {
    sizes.push_back(flat_size);
  }
  return DynamicReshape(input, sizes);
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
  int64_t size1 = xla::ShapeUtil::ByteSizeOfPrimitiveType(type1);
  int64_t size2 = xla::ShapeUtil::ByteSizeOfPrimitiveType(type2);
  if (xla::primitive_util::IsComplexType(type1)) {
    return (!xla::primitive_util::IsComplexType(type2) || size1 >= size2)
               ? type1
               : type2;
  }
  if (xla::primitive_util::IsComplexType(type2)) {
    return type2;
  }
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
    if (size1 > size2) {
      return type1;
    }
    if (size2 > size1) {
      return type2;
    }
    // At this point, they are not the same type, they are both integers, and
    // they have the same size. One of them must be unsigned and the other
    // signed, convert to unsigned.
    return xla::primitive_util::UnsignedIntegralTypeForBitWidth(
        xla::primitive_util::BitWidth(type1));
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

xla::PrimitiveType XlaHelpers::PromoteType(xla::PrimitiveType type1,
                                           xla::PrimitiveType type2,
                                           xla::PrimitiveType type3) {
  return PromoteType(PromoteType(type1, type2), type3);
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
  xla::PrimitiveType result_type = PromoteType(type1, type2, type3);
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

xla::Shape XlaHelpers::GetPromotedShape(const xla::Shape& shape1,
                                        const xla::Shape& shape2) {
  return xla::ShapeUtil::MakeShape(
      shape1.element_type(),
      torch::lazy::GetPromotedShape(
          xla::util::ToVector<int64_t>(shape1.dimensions()),
          xla::util::ToVector<int64_t>(shape2.dimensions())));
}

xla::Shape XlaHelpers::GetPromotedBinaryOpShape(const xla::Shape& shape1,
                                                const xla::Shape& shape2) {
  return xla::ShapeUtil::MakeShape(
      PromoteType(shape1.element_type(), shape2.element_type()),
      torch::lazy::GetPromotedShape(
          xla::util::ToVector<int64_t>(shape1.dimensions()),
          xla::util::ToVector<int64_t>(shape2.dimensions())));
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
  int64_t size_delta = shape_dims.size() - op_shape_dims.size();
  xla::XlaOp new_op = op;
  if (!std::equal(op_shape_dims.begin(), op_shape_dims.end(),
                  shape_dims.begin() + size_delta)) {
    // If the base N dimensions do not match, broadcast the original op.
    // Example:
    //   op_shape =       [3, 1, 5]
    //   shape    = [6, 8, 3, 4, 5]
    // After this operation we will have:
    //   op_shape =       [3, 4, 5]
    std::vector<int64_t> common_shape_dims(shape_dims.begin() + size_delta,
                                           shape_dims.end());
    std::vector<int64_t> broadcast_dimensions(op_shape_dims.size());
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
    std::vector<int64_t> broadcast_sizes(shape_dims.begin(),
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

xla::XlaOp XlaHelpers::PromotedLogicalBinaryOp(
    xla::XlaOp op1, xla::XlaOp op2,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& bin_op) {
  // XLA only supports bitwise_and/or/xor so we need to cast inputs to
  // PRED first.
  op1 = xla::ConvertElementType(op1, xla::PrimitiveType::PRED);
  op2 = xla::ConvertElementType(op2, xla::PrimitiveType::PRED);
  return bin_op(op1, op2);
}

xla::XlaOp XlaHelpers::PromotedLogicalUnaryOp(
    xla::XlaOp op, const std::function<xla::XlaOp(xla::XlaOp)>& unary_op) {
  // XLA only supports bitwise_not so we need to cast inputs to
  // PRED first.
  op = xla::ConvertElementType(op, xla::PrimitiveType::PRED);
  return unary_op(op);
}

xla::StatusOr<xla::XlaComputation> XlaHelpers::WrapXlaComputation(
    const xla::XlaComputation& computation,
    const std::vector<xla::Shape>& parameter_shapes,
    std::vector<std::pair<int64_t, int64_t>> input_output_alias_pair) {
  xla::XlaBuilder builder(computation.proto().name());

  // Construct a single tuple parameter.
  xla::Shape input_tuple_shape;
  input_tuple_shape.set_element_type(xla::PrimitiveType::TUPLE);
  input_tuple_shape.mutable_tuple_shapes()->reserve(parameter_shapes.size());
  for (int i = 0; i < parameter_shapes.size(); ++i) {
    *input_tuple_shape.add_tuple_shapes() = parameter_shapes[i];
  }
  xla::XlaOp input_tuple = xla::Parameter(&builder, 0, input_tuple_shape, "in");

  // Handle the results of the original computation.
  std::vector<xla::XlaOp> inner_params;
  inner_params.reserve(parameter_shapes.size());
  for (int i = 0; i < parameter_shapes.size(); ++i) {
    inner_params.push_back(xla::GetTupleElement(input_tuple, i));
  }

  // Call the original computation.
  xla::XlaOp orig_result = xla::Call(&builder, computation, inner_params);

  // Rebuild aliasing.
  for (const auto& [input_index, output_index] : input_output_alias_pair) {
    // Both input and output will be a tuple so parameter_number will always be
    // 0
    builder.SetUpAlias(/*output_index=*/xla::ShapeIndex({output_index}),
                       /*param_number=*/0,
                       /*param_index=*/xla::ShapeIndex({input_index}));
  }

  return builder.Build(orig_result);
}

torch::lazy::Shape XlaHelpers::ConvertXlaShapeToLazy(const xla::Shape& shape) {
  at::ScalarType scalar_type = TensorTypeFromXlaType(shape.element_type());
  c10::optional<std::vector<bool>> is_symbolic = c10::nullopt;
  if (shape.is_dynamic()) {
    std::vector<bool> xla_dynamic_dimensions =
        xla::util::ToVector<bool>(shape.dynamic_dimensions());
    is_symbolic = c10::make_optional(xla_dynamic_dimensions);
  }

  return torch::lazy::Shape(scalar_type,
                            xla::util::ToVector<int64_t>(shape.dimensions()),
                            std::move(is_symbolic));
}

}  // namespace torch_xla
