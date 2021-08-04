#include "lazy_xla/csrc/compiler/helpers.h"

#include <limits>

#include "absl/strings/str_join.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/ltc_logging.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_xla/csrc/compiler/convert_ops.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_lazy_tensors {
namespace compiler {
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

bool XlaHelpers::SameStaticDimensions(const xla::Shape& shape1,
                                      const xla::Shape& shape2) {
  return shape1.is_static() && shape2.is_static() &&
         shape1.dimensions() == shape2.dimensions();
}

xla::Shape XlaHelpers::XlaShape(const lazy_tensors::Shape& shape) {
  if (shape.element_type() == lazy_tensors::PrimitiveType::TUPLE) {
    std::vector<xla::Shape> shapes;
    for (const lazy_tensors::Shape& element_shape : shape.tuple_shapes()) {
      shapes.push_back(XlaShape(element_shape));
    }
    return xla::ShapeUtil::MakeTupleShape(shapes);
  }
  return xla::ShapeUtil::MakeShapeWithLayout(
      xla::ComputationClient::XlaPrimitiveType(shape.element_type()),
      shape.dimensions(), shape.layout().minor_to_major());
}

lazy_tensors::Shape XlaHelpers::LazyTensorsShape(const xla::Shape& shape) {
  if (shape.element_type() == xla::PrimitiveType::TUPLE) {
    std::vector<lazy_tensors::Shape> shapes;
    for (const xla::Shape& element_shape : shape.tuple_shapes()) {
      shapes.push_back(LazyTensorsShape(element_shape));
    }
    return lazy_tensors::ShapeUtil::MakeTupleShape(shapes);
  }
  lazy_tensors::Shape lazy_shape(
      xla::ComputationClient::LazyTensorPrimitiveType(shape.element_type()),
      shape.dimensions());
  lazy_tensors::Layout layout;
  for (const int64_t dim_index : shape.layout().minor_to_major()) {
    layout.add_minor_to_major(dim_index);
  }
  *lazy_shape.mutable_layout() = layout;
  return lazy_shape;
}

xla::XlaOp XlaHelpers::BroadcastDimensions(
    xla::XlaOp input, absl::Span<const xla::int64> dimensions,
    absl::Span<const xla::int64> sizes) {
  LTC_CHECK_EQ(dimensions.size(), sizes.size());
  std::vector<xla::int64> bcast_sizes = SizesOfXlaOp(input);
  for (size_t i = 0; i < dimensions.size(); ++i) {
    bcast_sizes.at(dimensions[i]) = sizes[i];
  }
  return xla::BroadcastInDim(
      input, bcast_sizes,
      torch_lazy_tensors::Helpers::GetAllDimensions(bcast_sizes.size()));
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

XlaHelpers::DynamicSize XlaHelpers::GetDimensionsSize(
    absl::Span<const xla::XlaOp> inputs,
    absl::Span<const xla::int64> dimensions) {
  LTC_CHECK(!inputs.empty());
  xla::PrimitiveType size_type = xla::ComputationClient::XlaPrimitiveType(
      GetShapeDimensionType(/*device=*/nullptr));
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
  LTC_CHECK_LE(offset + shape.rank(), expected_rank);
  if (shape.rank() == expected_rank) {
    return input;
  }
  std::vector<xla::int64> dimensions(expected_rank - offset - shape.rank(), 1);
  dimensions.insert(dimensions.end(), shape.dimensions().begin(),
                    shape.dimensions().end());
  dimensions.insert(dimensions.end(), offset, 1);
  return xla::Reshape(input, dimensions);
}

xla::XlaOp XlaHelpers::DynamicReshape(
    xla::XlaOp input, absl::Span<const xla::int64> output_sizes) {
  const xla::Shape& input_shape = ShapeOfXlaOp(input);
  if (output_sizes == input_shape.dimensions()) {
    return input;
  }
  auto info = torch_lazy_tensors::Helpers::GetDynamicReshapeInfo(
      LazyTensorsShape(input_shape), output_sizes);
  if (info) {
    return xla::ReshapeWithInferredDimension(input, output_sizes,
                                             info->dynamic_dimension);
  }
  return xla::Reshape(input, output_sizes);
}

xla::XlaOp XlaHelpers::DynamicReshapeAs(xla::XlaOp input,
                                        const xla::Shape& shape) {
  const xla::Shape& input_shape = ShapeOfXlaOp(input);
  xla::int64 dynamic_dimension =
      torch_lazy_tensors::Helpers::GetDynamicDimension(LazyTensorsShape(shape));
  if (dynamic_dimension >= 0) {
    return xla::ReshapeWithInferredDimension(input, shape.dimensions(),
                                             dynamic_dimension);
  }
  return shape.dimensions() == input_shape.dimensions()
             ? input
             : xla::Reshape(input, shape.dimensions());
}

xla::XlaOp XlaHelpers::Flatten(xla::XlaOp input, xla::Shape* input_shape) {
  lazy_tensors::util::MaybePtr<xla::Shape> input_shape_tmp(input_shape);
  *input_shape_tmp = ShapeOfXlaOp(input);
  if (input_shape_tmp->rank() == 1) {
    return input;
  }
  xla::int64 input_elements = xla::ShapeUtil::ElementsIn(*input_shape_tmp);
  return DynamicReshape(input, {input_elements});
}

xla::XlaOp XlaHelpers::FlattenDimRange(xla::XlaOp input, xla::int64 start,
                                       xla::int64 range,
                                       xla::Shape* input_shape) {
  lazy_tensors::util::MaybePtr<xla::Shape> input_shape_tmp(input_shape);
  *input_shape_tmp = ShapeOfXlaOp(input);

  std::vector<xla::int64> sizes;
  xla::int64 flat_size = -1;
  for (xla::int64 dim = 0; dim < input_shape_tmp->rank(); ++dim) {
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

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteValues(xla::XlaOp op1,
                                                            xla::XlaOp op2) {
  xla::PrimitiveType type1 = TypeOfXlaOp(op1);
  xla::PrimitiveType type2 = TypeOfXlaOp(op2);
  xla::PrimitiveType result_type = xla::ComputationClient::XlaPrimitiveType(
      torch_lazy_tensors::Helpers::PromoteType(
          xla::ComputationClient::LazyTensorPrimitiveType(type1),
          xla::ComputationClient::LazyTensorPrimitiveType(type2)));
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
  xla::PrimitiveType result_type = xla::ComputationClient::XlaPrimitiveType(
      torch_lazy_tensors::Helpers::PromoteType(
          torch_lazy_tensors::Helpers::PromoteType(
              xla::ComputationClient::LazyTensorPrimitiveType(type1),
              xla::ComputationClient::LazyTensorPrimitiveType(type2)),
          xla::ComputationClient::LazyTensorPrimitiveType(type3)));
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

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteShapes(xla::XlaOp op1,
                                                            xla::XlaOp op2) {
  const xla::Shape& shape1 = ShapeOfXlaOp(op1);
  const xla::Shape& shape2 = ShapeOfXlaOp(op2);
  if (xla::ShapeUtil::Compatible(shape1, shape2)) {
    // Fast path shortcut if the shapes already matches in dimensions.
    return std::pair<xla::XlaOp, xla::XlaOp>(op1, op2);
  }
  LTC_CHECK(xla::ShapeUtil::SameElementType(shape1, shape2))
      << shape1 << " and " << shape2;

  xla::Shape shape = XlaShape(torch_lazy_tensors::Helpers::GetPromotedShape(
      LazyTensorsShape(shape1), LazyTensorsShape(shape2)));
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
  LTC_CHECK_GE(shape_dims.size(), op_shape_dims.size())
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

xla::XlaOp XlaHelpers::PromotedLogicalBinaryOp(
    xla::XlaOp op1, xla::XlaOp op2,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& bin_op) {
  // XLA only supports bitwise_and/or/xor so we need to cast inputs to
  // PRED first.
  op1 = xla::ConvertElementType(op1, xla::PrimitiveType::PRED);
  op2 = xla::ConvertElementType(op2, xla::PrimitiveType::PRED);
  return bin_op(op1, op2);
}

xla::PaddingConfig XlaHelpers::MakeXlaPaddingConfigFromNdPadding(
    absl::Span<const xla::int64> padding) {
  LTC_CHECK_EQ(padding.size() % 2, 0)
      << "Padding specification must have even length";
  LTC_CHECK(!padding.empty()) << "Padding specification cannot be empty";
  xla::PaddingConfig padding_config;
  for (int i = 0; i < padding.size(); i += 2) {
    xla::PaddingConfig::PaddingConfigDimension* dims =
        padding_config.add_dimensions();
    dims->set_edge_padding_low(padding[padding.size() - i - 2]);
    dims->set_edge_padding_high(padding[padding.size() - i - 1]);
  }
  return padding_config;
}

lazy_tensors::PrimitiveType XlaHelpers::LazyTensorPrimitiveType(
    xla::PrimitiveType type) {
  return xla::ComputationClient::LazyTensorPrimitiveType(type);
}

}  // namespace compiler
}  // namespace torch_lazy_tensors
