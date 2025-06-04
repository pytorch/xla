#include "torch_xla/csrc/helpers.h"

#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/util.h>

#include <iterator>
#include <limits>

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/dtype.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch_xla/csrc/runtime/util.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/tensor_util.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"

namespace torch_xla {
namespace {

xla::XlaOp ConvertBinaryOpResult(xla::XlaOp op1, xla::XlaOp op2,
                                 xla::XlaOp result) {
  xla::PrimitiveType type1 = XlaHelpers::TypeOfXlaOp(op1);
  xla::PrimitiveType type2 = XlaHelpers::TypeOfXlaOp(op2);
  xla::PrimitiveType result_type = XlaHelpers::TypeOfXlaOp(result);
  if (type1 == type2 && type1 != result_type) {
    return ConvertTo(result, result_type, type1);
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

xla::XlaComputation CreateMinMaxComputation(const std::string& name,
                                            xla::PrimitiveType value_type,
                                            xla::PrimitiveType index_type,
                                            bool is_min) {
  xla::XlaBuilder builder(name);
  xla::XlaOp lhs_value = xla::Parameter(
      &builder, 0, xla::ShapeUtil::MakeShape(value_type, {}), "lhs_value");
  xla::XlaOp lhs_index = xla::Parameter(
      &builder, 1, xla::ShapeUtil::MakeShape(index_type, {}), "lhs_index");
  xla::XlaOp rhs_value = xla::Parameter(
      &builder, 2, xla::ShapeUtil::MakeShape(value_type, {}), "rhs_value");
  xla::XlaOp rhs_index = xla::Parameter(
      &builder, 3, xla::ShapeUtil::MakeShape(index_type, {}), "rhs_index");

  xla::XlaOp cmp =
      is_min ? xla::Le(lhs_value, rhs_value) : xla::Ge(lhs_value, rhs_value);
  xla::XlaOp max = xla::Select(cmp, lhs_value, rhs_value);
  xla::XlaOp arg_max = xla::Select(cmp, lhs_index, rhs_index);
  xla::XlaOp eq = xla::Eq(lhs_value, rhs_value);
  xla::XlaOp tie_id = xla::Min(lhs_index, rhs_index);
  arg_max = xla::Select(eq, tie_id, arg_max);
  xla::Tuple(&builder, {max, arg_max});
  return ConsumeValue(builder.Build());
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
    if (XlaHelpers::IsUnboundedDynamismEnabled()) {
      XLA_CHECK(sizes[i] != xla::Shape::kUnboundedSize);
    }
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
  for (int64_t i = 0; i < shape.dimensions_size(); ++i) {
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
    const xla::Shape& shape = ShapeHelper::ShapeOfXlaOp(input);
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

xla::XlaComputation XlaHelpers::CreateMaxAndArgMaxComputation(
    xla::PrimitiveType value_type, xla::PrimitiveType index_type) {
  return CreateMinMaxComputation("MaxAndArgMaxComputation", value_type,
                                 index_type, /*is_min=*/false);
}

std::vector<int64_t> XlaHelpers::SizesOfXlaOp(xla::XlaOp op) {
  const xla::Shape& op_shape = ShapeHelper::ShapeOfXlaOp(op);
  return std::vector<int64_t>(op_shape.dimensions().begin(),
                              op_shape.dimensions().end());
}

xla::PrimitiveType XlaHelpers::TypeOfXlaOp(xla::XlaOp op) {
  return ShapeHelper::ShapeOfXlaOp(op).element_type();
}

xla::XlaOp XlaHelpers::ReshapeToRank(xla::XlaOp input, int64_t expected_rank,
                                     int64_t offset) {
  const xla::Shape& shape = ShapeHelper::ShapeOfXlaOp(input);
  XLA_CHECK_LE(offset + shape.dimensions_size(), expected_rank);
  if (shape.dimensions_size() == expected_rank) {
    return input;
  }
  std::vector<int64_t> dimensions(
      expected_rank - offset - shape.dimensions_size(), 1);
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
  if (info.output_shape.dimensions_size() > 0) {
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
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  bool is_output_sizes_unbounded_dynamic = std::any_of(
      output_sizes.begin(), output_sizes.end(),
      [](int64_t size) { return size == xla::Shape::kUnboundedSize; });
  XLA_CHECK(!is_output_sizes_unbounded_dynamic)
      << "reshape operation does not support unbounded dynamic output shape.";
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
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  int64_t dynamic_dimension = GetDynamicDimension(shape);
  if (dynamic_dimension >= 0) {
    return xla::ReshapeWithInferredDimension(input, shape.dimensions(),
                                             dynamic_dimension);
  }
  return shape.dimensions() == input_shape.dimensions()
             ? input
             : xla::Reshape(input, shape.dimensions());
}

bool XlaHelpers::IsUnboundedDynamic(const xla::Shape& shape) {
  XLA_CHECK(XlaHelpers::IsUnboundedDynamismEnabled())
      << "set EXPERIMENTAL_XLA_UNBOUNDED_DYNAMISM=1 to run any unbounded "
         "dynamism workload.";
  const absl::Span<const int64_t> dims = shape.dimensions();
  return std::any_of(dims.begin(), dims.end(), [](int64_t size) {
    return size == xla::Shape::kUnboundedSize;
  });
}

xla::XlaOp XlaHelpers::DynamicUnboundedReshape(
    xla::XlaOp input, xla::XlaOp aux_input,
    absl::Span<const int64_t> output_sizes) {
  XLA_CHECK(XlaHelpers::IsUnboundedDynamismEnabled())
      << "set EXPERIMENTAL_XLA_UNBOUNDED_DYNAMISM=1 to run any unbounded "
         "dynamism workload.";
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  bool output_sizes_unbounded_dynamic = std::any_of(
      output_sizes.begin(), output_sizes.end(),
      [](int64_t size) { return size == xla::Shape::kUnboundedSize; });
  XLA_CHECK(input_shape.is_unbounded_dynamic() ||
            output_sizes_unbounded_dynamic)
      << "XlaHelpers::DynamicUnboundedReshape constrainled failed! At least "
         "one of input and output size needs to be unbounded dynamic.";
  if (input_shape.is_unbounded_dynamic() && !output_sizes_unbounded_dynamic) {
    return xla::Reshape(input, output_sizes);
  }
  const xla::Shape& aux_input_shape = ShapeHelper::ShapeOfXlaOp(aux_input);
  XLA_CHECK(output_sizes.size() == aux_input_shape.dimensions_size())
      << "XlaHelpers::DynamicUnboundedReshape constrainled failed! output size"
         " and aux_input should have same rank.";

  std::vector<xla::XlaOp> get_dim_ops;
  std::vector<xla::XlaOp> reshaped_ops;
  std::vector<bool> output_dynamic(output_sizes.size(), false);

  for (int i = 0; i < output_sizes.size(); i++) {
    if (output_sizes[i] == xla::Shape::kUnboundedSize) {
      output_dynamic[i] = true;
      get_dim_ops.push_back(xla::GetDimensionSize(aux_input, i));
    } else {
      get_dim_ops.push_back(XlaHelpers::ScalarValue<int32_t>(
          output_sizes[i], aux_input.builder()));
    }
  }

  // Create the reshape from scalar to 1-D vector
  for (auto get_dim_op : get_dim_ops) {
    reshaped_ops.push_back(xla::Reshape(get_dim_op, {1}));
  }

  // Create Concatenate op
  auto concat_op = xla::ConcatInDim(input.builder(), reshaped_ops, {0});
  return xla::CustomCall(
      aux_input.builder(), "mhlo.dynamic_reshape", {input, concat_op},
      xla::ShapeUtil::MakeShape(aux_input_shape.element_type(), output_sizes,
                                output_dynamic));
}

bool XlaHelpers::SameStaticDimensions(const xla::Shape& shape1,
                                      const xla::Shape& shape2) {
  return shape1.is_static() && shape2.is_static() &&
         shape1.dimensions() == shape2.dimensions();
}

xla::XlaOp XlaHelpers::Flatten(xla::XlaOp input, xla::Shape* input_shape) {
  runtime::util::MaybePtr<xla::Shape> input_shape_tmp(input_shape);
  *input_shape_tmp = ShapeHelper::ShapeOfXlaOp(input);
  if (input_shape_tmp->dimensions_size() == 1) {
    return input;
  }
  int64_t input_elements = xla::ShapeUtil::ElementsIn(*input_shape_tmp);
  return DynamicReshape(input, {input_elements});
}

xla::XlaOp XlaHelpers::FlattenDimRange(xla::XlaOp input, int64_t start,
                                       int64_t range, xla::Shape* input_shape) {
  runtime::util::MaybePtr<xla::Shape> input_shape_tmp(input_shape);
  *input_shape_tmp = ShapeHelper::ShapeOfXlaOp(input);

  std::vector<int64_t> sizes;
  int64_t flat_size = -1;
  for (int64_t dim = 0; dim < input_shape_tmp->dimensions_size(); ++dim) {
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
  const xla::Shape& shape = ShapeHelper::ShapeOfXlaOp(value0);
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
    op1 = ConvertTo(op1, type1, result_type);
  }
  if (type2 != result_type) {
    op2 = ConvertTo(op2, type2, result_type);
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
    op1 = ConvertTo(op1, type1, result_type);
  }
  if (type2 != result_type) {
    op2 = ConvertTo(op2, type2, result_type);
  }
  if (type3 != result_type) {
    op3 = ConvertTo(op3, type3, result_type);
  }
  return std::tuple<xla::XlaOp, xla::XlaOp, xla::XlaOp>(op1, op2, op3);
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteSecondValue(
    xla::XlaOp op1, xla::XlaOp op2) {
  xla::PrimitiveType type1 = TypeOfXlaOp(op1);
  xla::PrimitiveType type2 = TypeOfXlaOp(op2);
  return type1 == type2 ? std::pair<xla::XlaOp, xla::XlaOp>(op1, op2)
                        : std::pair<xla::XlaOp, xla::XlaOp>(
                              op1, ConvertTo(op2, type2, type1));
}

namespace {

// Polulate 'dimensions' and 'dynamic_dimension' from left-most dimensions of
// 'shape' respective to 'count'.
void ExtractDimensionSizesAndDynamicDimensionsFromShape(
    const xla::Shape& shape, int64_t count, std::vector<int64_t>& dimensions,
    std::vector<bool>& dynamic_dimensions) {
  dimensions.insert(dimensions.end(), shape.dimensions().begin(),
                    shape.dimensions().begin() + count);
  dynamic_dimensions.insert(dynamic_dimensions.end(),
                            shape.dynamic_dimensions().begin(),
                            shape.dynamic_dimensions().begin() + count);
}

}  // namespace

xla::Shape XlaHelpers::GetPromotedShape(const xla::Shape& shape1,
                                        const xla::Shape& shape2) {
  std::vector<int64_t> dimensions;
  std::vector<bool> dynamic_dimensions;

  // If the rank of a shape is bigger than then other, fill up the first
  // dimensions with the ones of the bigger.
  // Example:
  //   shape1 = [9, ?, 6, ?, ?]
  //   shape2 =       [6, 1, 2]
  // Insert [9, ?] into the dimensions vector.
  if (shape1.dimensions().size() > shape2.dimensions().size())
    ExtractDimensionSizesAndDynamicDimensionsFromShape(
        shape1, shape1.dimensions().size() - shape2.dimensions().size(),
        dimensions, dynamic_dimensions);
  else
    ExtractDimensionSizesAndDynamicDimensionsFromShape(
        shape2, shape2.dimensions().size() - shape1.dimensions().size(),
        dimensions, dynamic_dimensions);

  // For the common dimensions, they must match, or one of them be 1.
  size_t min_size =
      std::min(shape1.dimensions().size(), shape2.dimensions().size());
  for (size_t i = 0; i < min_size; i++) {
    int64_t dim1 =
        shape1.dimensions()[shape1.dimensions().size() - min_size + i];
    int64_t dynamic_dim1 =
        shape1.dynamic_dimensions()[shape1.dynamic_dimensions().size() -
                                    min_size + i];
    int64_t dim2 =
        shape2.dimensions()[shape2.dimensions().size() - min_size + i];
    int64_t dynamic_dim2 =
        shape2.dynamic_dimensions()[shape2.dynamic_dimensions().size() -
                                    min_size + i];

    XLA_CHECK(dim1 == dim2 || dim1 == 1 || dim2 == 1 ||
              dim1 == xla::Shape::kUnboundedSize ||
              dim2 == xla::Shape::kUnboundedSize);

    // TODO: Consider replacing the broadcasting logic below with
    // 'xla::ShapeInference::InferDegenerateDimensionBroadcastShape' resuing the
    // existing xla shape inference machinery. The function is better suited for
    // inferring shape when the participating
    // ranks are the same. However, the function has private visisbilty and
    // needs some refactoring before use.
    if (dim1 == 1 || dim2 == 1) {
      // dim1 | dim2 | result
      // 1   | X   | X
      // 1   | <=X | <=X
      // 1   | ?   | ?
      // X   | 1   | X
      // <=X | 1   | <=X
      // ?   | 1   | ?
      dimensions.push_back(dim1 == 1 ? dim2 : dim1);
      dynamic_dimensions.push_back(dim1 == 1 ? dynamic_dim2 : dynamic_dim1);
    } else if (dim1 == dim2) {
      // dim1 | dim2 | result
      // X   | X   | X
      // X   | <=X | <=X
      // <=X | X   | <=X
      // <=X | <=X | <=X
      // ?   | ?   | ?
      dimensions.push_back(dim1);
      dynamic_dimensions.push_back(dynamic_dim1 || dynamic_dim2);
    } else {
      // dim1 == xla::Shape::kUnboundedSize || dim2 ==
      // xla::Shape::kUnboundedSize
      //
      // dim1 | dim2 | result
      // X   | ?   | X
      // ?   | X   | X
      // <=X | ?   | ?
      // ?   | <=X | ?
      dimensions.push_back(dim1 == xla::Shape::kUnboundedSize ? dim2 : dim1);
      dynamic_dimensions.push_back(
          dim1 == xla::Shape::kUnboundedSize ? dynamic_dim2 : dynamic_dim1);
    }
  }
  return xla::ShapeUtil::MakeShape(shape1.element_type(), dimensions,
                                   dynamic_dimensions);
}

std::vector<int64_t> XlaHelpers::getBroadcastDimensions(xla::XlaOp op1,
                                                        xla::XlaOp op2) {
  const xla::Shape& shape1 = ShapeHelper::ShapeOfXlaOp(op1);
  const xla::Shape& shape2 = ShapeHelper::ShapeOfXlaOp(op2);
  if (shape1.dimensions_size() == 0 || shape2.dimensions_size() == 0 ||
      shape1.dimensions_size() == shape2.dimensions_size())
    return {};

  std::vector<int64_t> broadcast_dimensions(shape1.dimensions_size() <=
                                                    shape2.dimensions_size()
                                                ? shape1.dimensions_size()
                                                : shape2.dimensions_size());
  std::iota(broadcast_dimensions.begin(), broadcast_dimensions.end(),
            std::abs(shape1.dimensions_size() - shape2.dimensions_size()));
  return broadcast_dimensions;
}

xla::Shape XlaHelpers::GetPromotedBinaryOpShape(const xla::Shape& shape1,
                                                const xla::Shape& shape2) {
  if (!shape1.is_dynamic() && !shape2.is_dynamic()) {
    auto promoted_shape = GetPromotedShape(shape1, shape2);
    return xla::ShapeUtil::MakeShape(
        PromoteType(shape1.element_type(), shape2.element_type()),
        promoted_shape.dimensions());
  }
  if (XlaHelpers::IsUnboundedDynamismEnabled()) {
    XLA_CHECK(!XlaHelpers::IsUnboundedDynamic(shape1) &&
              !XlaHelpers::IsUnboundedDynamic(shape2))
        << "Unreachable for unbounded dynamic code\n";
  }
  return GetPromotedDynamicShape(shape1, shape2);
}

xla::Shape XlaHelpers::GetPromotedDynamicShape(const xla::Shape& shape1,
                                               const xla::Shape& shape2) {
  std::vector<int64_t> upper_bounds1 =
      runtime::util::ToVector<int64_t>(shape1.dimensions());
  std::vector<int64_t> upper_bounds2 =
      runtime::util::ToVector<int64_t>(shape2.dimensions());
  absl::Span<const bool> dyn_dims1 = shape1.dynamic_dimensions();
  absl::Span<const bool> dyn_dims2 = shape2.dynamic_dimensions();
  std::vector<int64_t> upper_bounds;
  std::vector<bool> dyn_dims;

  // See
  // https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
  if (upper_bounds1.size() > upper_bounds2.size()) {
    upper_bounds.insert(
        upper_bounds.end(), upper_bounds1.begin(),
        upper_bounds1.begin() + (upper_bounds1.size() - upper_bounds2.size()));
    dyn_dims.insert(dyn_dims.end(), dyn_dims1.begin(),
                    dyn_dims1.begin() + (dyn_dims1.size() - dyn_dims2.size()));
  } else {
    upper_bounds.insert(
        upper_bounds.end(), upper_bounds2.begin(),
        upper_bounds2.begin() + (upper_bounds2.size() - upper_bounds1.size()));
    dyn_dims.insert(dyn_dims.end(), dyn_dims2.begin(),
                    dyn_dims2.begin() + (dyn_dims2.size() - dyn_dims1.size()));
  }
  size_t min_size = std::min(upper_bounds1.size(), upper_bounds2.size());
  for (const auto i : c10::irange(min_size)) {
    int64_t ubound1 = upper_bounds1[upper_bounds1.size() - min_size + i];
    int64_t ubound2 = upper_bounds2[upper_bounds2.size() - min_size + i];
    bool is_dim1_dynamic = dyn_dims1[dyn_dims1.size() - min_size + i];
    bool is_dim2_dynamic = dyn_dims2[dyn_dims2.size() - min_size + i];
    if (!is_dim1_dynamic && !is_dim2_dynamic) {
      XLA_CHECK(ubound1 == 1 || ubound2 == 1 || ubound1 == ubound2)
          << "At dimension " << i
          << ", both dimension are static with real size " << ubound1 << " and "
          << ubound2;
    } else {
      // For now, if both dimension are dynamic and has the same upper bound,
      // we regard this dimension to be broadcastable.
      XLA_CHECK((is_dim1_dynamic && !is_dim2_dynamic && ubound2 == 1) ||
                (is_dim2_dynamic && !is_dim1_dynamic && ubound1 == 1) ||
                (is_dim1_dynamic && is_dim2_dynamic && ubound1 == ubound2))
          << "At dimension " << i << ", operand1 has dimension size " << ubound1
          << " isDynamic=" << is_dim1_dynamic
          << " vs operand2 has dimension size " << ubound2
          << " isDynamic=" << is_dim2_dynamic;
    }

    int64_t ubound = std::max<int64_t>(ubound1, ubound2);
    upper_bounds.push_back(ubound);
    bool is_dim_dynamic = is_dim1_dynamic || is_dim2_dynamic;
    dyn_dims.push_back(is_dim_dynamic);
  }
  const xla::Shape& promoted_shape = xla::ShapeUtil::MakeShape(
      PromoteType(shape1.element_type(), shape2.element_type()), upper_bounds,
      dyn_dims);

  return promoted_shape;
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteShapes(xla::XlaOp op1,
                                                            xla::XlaOp op2) {
  const xla::Shape& shape1 = ShapeHelper::ShapeOfXlaOp(op1);
  const xla::Shape& shape2 = ShapeHelper::ShapeOfXlaOp(op2);

  xla::Shape shape = GetPromotedShape(shape1, shape2);
  if (shape1.is_unbounded_dynamic() || shape2.is_unbounded_dynamic()) {
    return ImplicitBroadcastWithUnboundedDynamicShapes(op1, op2, shape);
  }

  XLA_CHECK(xla::ShapeUtil::SameElementType(shape1, shape2))
      << shape1 << " and " << shape2;

  return std::pair<xla::XlaOp, xla::XlaOp>(op1, op2);
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

namespace {

// Extract the 'num_dims' counts of dimension sizes from the 'op', reshape
// them to 'tensor<1xi32>', and append them at 'op_dims' after prepending
// 'pad_count' of 1's reshaped to 'tensor<1xi32>'.
std::vector<xla::XlaOp> ExtractDimensionSizesWithPadding(const xla::XlaOp op,
                                                         size_t num_dims,
                                                         int pad_count) {
  std::vector<xla::XlaOp> op_dims;
  for (size_t i = 0; i < pad_count; i++) {
    op_dims.push_back(
        xla::Reshape(XlaHelpers::ScalarValue<int32_t>(1, op.builder()), {1}));
  }

  for (size_t i = 0; i < num_dims; i++) {
    op_dims.push_back(xla::Reshape(xla::GetDimensionSize(op, i), {1}));
  }

  return op_dims;
}

// Stringify the broadcast dimensions to provide for the 'backend_config'
// attribute of the generated custom_call.
std::string StringifyBroadcastDimensions(std::vector<int64_t> broadcast_dims) {
  std::string str("{broadcast_dimensions=[");
  if (broadcast_dims.size() >= 1) {
    str += std::to_string(broadcast_dims[0]);
  }
  for (size_t i = 1; i < broadcast_dims.size(); i++) {
    str += ", " + std::to_string(broadcast_dims[i]);
  }
  str += "]}";
  return str;
};

}  // namespace

// Emit XLA custom call for dynamic_broadcast_in_dim.
xla::XlaOp XlaHelpers::DynamicBroadcastInDim(
    xla::XlaOp op, const xla::Shape& final_shape,
    xla::XlaOp final_broadcast_dimensions) {
  const xla::Shape& shape = ShapeHelper::ShapeOfXlaOp(op);

  std::vector<int64_t> op_broadcast_dims(shape.dimensions().size());
  std::iota(op_broadcast_dims.begin(), op_broadcast_dims.end(),
            final_shape.dimensions().size() - shape.dimensions().size());

  return xla::CustomCall(
      op.builder(), "mhlo.dynamic_broadcast_in_dim",
      /*operands=*/{op, final_broadcast_dimensions}, /*shape*/ final_shape,
      /*opaque=*/StringifyBroadcastDimensions(op_broadcast_dims));
}

xla::XlaOp XlaHelpers::DynamicUnboundedBroadcast(
    xla::XlaOp input, xla::XlaOp aux_input,
    const std::vector<int64_t>& aux_input_dimensions) {
  XLA_CHECK(XlaHelpers::IsUnboundedDynamismEnabled())
      << "set EXPERIMENTAL_XLA_UNBOUNDED_DYNAMISM=1 to run any unbounded "
         "dynamism workload.";

  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  const xla::Shape& aux_input_shape = ShapeHelper::ShapeOfXlaOp(aux_input);
  std::vector<int64_t> output_dimensions;
  std::vector<bool> output_dynamic;
  // Collect the dimension sizes and dynamic dimensions corresponding to the
  // final broadcasted shape.
  for (auto dim : aux_input_dimensions) {
    output_dimensions.push_back(aux_input_shape.dimensions(dim));
    output_dynamic.push_back(aux_input_shape.is_dynamic_dimension(dim));
  }

  for (int dim = 0; dim < input_shape.dimensions_size(); dim++) {
    output_dimensions.push_back(input_shape.dimensions(dim));
    output_dynamic.push_back(input_shape.is_dynamic_dimension(dim));
  }

  std::vector<xla::XlaOp> get_dim_ops;
  // Create a concatente op with the dimension sizes to be fed as the second
  // argument of the 'soon-to-be-created' dynamic_broadcast_in_dim.
  for (auto dim : aux_input_dimensions) {
    if (aux_input_shape.dimensions(dim) != xla::Shape::kUnboundedSize) {
      get_dim_ops.push_back(xla::Reshape(
          XlaHelpers::ScalarValue<int32_t>(aux_input_shape.dimensions(dim),
                                           aux_input.builder()),
          {1}));
    } else {
      get_dim_ops.push_back(
          xla::Reshape(xla::GetDimensionSize(aux_input, dim), {1}));
    }
  }

  for (int dim = 0; dim < input_shape.dimensions_size(); dim++) {
    if (input_shape.dimensions(dim) != xla::Shape::kUnboundedSize) {
      get_dim_ops.push_back(
          xla::Reshape(XlaHelpers::ScalarValue<int32_t>(
                           input_shape.dimensions(dim), input.builder()),
                       {1}));
    } else {
      get_dim_ops.push_back(
          xla::Reshape(xla::GetDimensionSize(input, dim), {1}));
    }
  }

  // Create Concatenate op
  auto concat_op = xla::ConcatInDim(input.builder(), get_dim_ops, {0});

  xla::Shape final_shape = xla::ShapeUtil::MakeShape(
      input_shape.element_type(), output_dimensions, output_dynamic);
  return DynamicBroadcastInDim(input, final_shape, concat_op);
}

std::pair<xla::XlaOp, xla::XlaOp>
XlaHelpers::ImplicitBroadcastWithUnboundedDynamicShapes(
    xla::XlaOp op1, xla::XlaOp op2, const xla::Shape& shape) {
  const xla::Shape& shape1 = ShapeHelper::ShapeOfXlaOp(op1);
  const xla::Shape& shape2 = ShapeHelper::ShapeOfXlaOp(op2);

  XLA_CHECK(shape1.is_unbounded_dynamic() || shape2.is_unbounded_dynamic());
  XLA_CHECK(shape.dimensions().size() ==
            std::max(shape1.dimensions().size(), shape2.dimensions().size()));

  // Collect the dimension sizes of the 'op1' and 'op2' in 'op1_dims' and
  // 'op2_dims' resp. with potential padding.
  //   Example:
  //     shape1 = [9, ?, 6, ?, ?]
  //     shape2 =       [6, 1, 2]
  //      shape = [9, ?, 6, ?, 2] where ?: represents unbounded dynamic size.
  //
  //   rank(shape1) = rank(shape): No padding needed and
  //   the pre-broadcast result, 'op1_dims', just collects the dimension sizes
  //   of shape1.
  //     op1_dims = [9, ?, 6, ?, ?]
  //   rank(shape2) < rank(shape): Make the rank of shape2 match
  //   that of shape by padding it with 1's. The pre-broadcast result is
  //   'op2_dims'.
  //     op2_dims = [1, 1, 6, 1, 2]
  std::vector<xla::XlaOp> op1_dims = ExtractDimensionSizesWithPadding(
      op1, shape1.dimensions().size(),
      shape.dimensions().size() - shape1.dimensions().size());
  std::vector<xla::XlaOp> op2_dims = ExtractDimensionSizesWithPadding(
      op2, shape2.dimensions().size(),
      shape.dimensions().size() - shape2.dimensions().size());

  // The broadcasted shape is the max of the individual pre-broadcasted
  // shapes. final_broadcast_dimensions = max(op1_dims, op2_dims).
  auto final_broadcast_dimensions =
      xla::Max(xla::ConcatInDim(op1.builder(), op1_dims, {0}),
               xla::ConcatInDim(op2.builder(), op2_dims, {0}));

  return std::pair<xla::XlaOp, xla::XlaOp>(
      DynamicBroadcastInDim(op1, shape, final_broadcast_dimensions),
      DynamicBroadcastInDim(op2, shape, final_broadcast_dimensions));
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

absl::StatusOr<xla::XlaComputation> XlaHelpers::WrapXlaComputation(
    const xla::XlaComputation& computation,
    const std::vector<xla::Shape>& parameter_shapes,
    const std::vector<xla::HloSharding>& parameter_shardings,
    const std::vector<size_t>& buffer_donor_indices) {
  xla::XlaBuilder builder(computation.proto().name());

  // Construct a single tuple parameter.
  xla::Shape input_tuple_shape;
  input_tuple_shape.set_element_type(xla::PrimitiveType::TUPLE);
  input_tuple_shape.mutable_tuple_shapes()->reserve(parameter_shapes.size());
  for (int i = 0; i < parameter_shapes.size(); ++i) {
    *input_tuple_shape.add_tuple_shapes() = parameter_shapes[i];
  }
  // Parameters are sharded, need to attach the tuple sharding to the wrapped
  // parameter.
  if (parameter_shardings.size()) {
    xla::HloSharding tuple_sharding =
        xla::HloSharding::Tuple(input_tuple_shape, parameter_shardings);
    builder.SetSharding(tuple_sharding.ToProto());
  }
  xla::XlaOp input_tuple = xla::Parameter(&builder, 0, input_tuple_shape, "in");
  if (parameter_shardings.size()) {
    builder.ClearSharding();
  }

  // Handle the results of the original computation.
  std::vector<xla::XlaOp> inner_params;
  inner_params.reserve(parameter_shapes.size());
  for (int i = 0; i < parameter_shapes.size(); ++i) {
    inner_params.push_back(xla::GetTupleElement(input_tuple, i));
  }

  // Call the original computation.
  xla::XlaOp orig_result = xla::Call(&builder, computation, inner_params);

  // Rebuild aliasing.
  for (const int64_t i : buffer_donor_indices) {
    builder.AddBufferDonor(/*param_number=*/0,
                           /*param_index=*/xla::ShapeIndex({i}));
  }

  return builder.Build(orig_result);
}

std::vector<xla::HloSharding> XlaHelpers::ExtractInputShardings(
    const xla::XlaComputation& computation) {
  std::vector<xla::HloSharding> param_shardings;
  // entry computation is always the last computation
  for (const xla::HloInstructionProto& instr :
       computation.proto().computations().rbegin()->instructions()) {
    if (instr.opcode() == HloOpcodeString(xla::HloOpcode::kParameter)) {
      const int64_t index = instr.parameter_number();
      // we assume that parameter is ordered.
      XLA_CHECK_EQ(index, param_shardings.size());
      param_shardings.push_back(*xla::HloSharding::FromProto(instr.sharding()));
      TF_VLOG(5) << "index = " << index
                 << " shape = " << xla::Shape(instr.shape()).ToString()
                 << " sharding = "
                 << xla::HloSharding::FromProto(instr.sharding())->ToString();
    }
  }
  return param_shardings;
}

torch::lazy::Shape XlaHelpers::ConvertXlaShapeToLazy(const xla::Shape& shape) {
  at::ScalarType scalar_type = MaybeUpcastToHostTorchType(shape.element_type());
  std::optional<std::vector<bool>> is_symbolic = std::nullopt;
  if (shape.is_dynamic()) {
    std::vector<bool> xla_dynamic_dimensions =
        runtime::util::ToVector<bool>(shape.dynamic_dimensions());
    is_symbolic = std::make_optional(xla_dynamic_dimensions);
  }

  return torch::lazy::Shape(
      scalar_type, runtime::util::ToVector<int64_t>(shape.dimensions()),
      std::move(is_symbolic));
}

}  // namespace torch_xla
