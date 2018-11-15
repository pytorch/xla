#include "helpers.h"

namespace torch {
namespace jit {

xla::PrecisionConfig XlaHelpers::BuildPrecisionConfig(
    const xla::PrecisionConfig::Precision conv_precision) {
  xla::PrecisionConfig precision_config;
  // Dot and convolution take two operators.
  precision_config.mutable_operand_precision()->Resize(
      /*new_size=*/2, conv_precision);
  return precision_config;
}

std::vector<int64_t> XlaHelpers::TensorDimensionSizes(const Value* tensor) {
  const auto tensor_type = tensor->type()->cast<CompleteTensorType>();
  CHECK(tensor_type);
  return tensor_type->sizes();
}

std::vector<xla::int64> XlaHelpers::I64List(const at::IntList& input) {
  std::vector<xla::int64> output(input.size());
  std::copy(input.begin(), input.end(), output.begin());
  return output;
}

xla::PaddingConfig XlaHelpers::MakeXlaPaddingConfig(
    const std::vector<int64_t>& padding) {
  xla::PaddingConfig padding_config;
  for (int i = 0; i < 2; ++i) {
    padding_config.add_dimensions();
  }
  for (int i = 0; i < 2; ++i) {
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(padding[i]);
    dims->set_edge_padding_high(padding[i]);
  }
  return padding_config;
}

xla::XlaComputation XlaHelpers::CreateAddComputation() {
  xla::XlaBuilder reduction_builder("xla_add_computation");
  const auto x = xla::Parameter(
      &reduction_builder, 0,
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {}), "x");
  const auto y = xla::Parameter(
      &reduction_builder, 1,
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {}), "y");
  Add(x, y);
  return reduction_builder.Build().ConsumeValueOrDie();
}

xla::PrimitiveType XlaHelpers::MakeXlaPrimitiveType(
    const at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::ScalarType::Float:
      return xla::PrimitiveType::F32;
    case at::ScalarType::Long:
      return xla::PrimitiveType::S64;
    default:
      LOG(FATAL) << "Type not supported: " << scalar_type;
  }
}

std::vector<xla::int64> XlaHelpers::ShapeSizes(const xla::Shape& shape) {
  std::vector<xla::int64> shape_sizes(shape.dimensions().begin(),
                                      shape.dimensions().end());
  return shape_sizes;
}

xla::Shape XlaHelpers::ShapeOfXlaOp(const xla::XlaOp& op) {
  return op.builder()->GetShape(op).ValueOrDie();
}

xla::PrimitiveType XlaHelpers::TypeOfXlaOp(const xla::XlaOp& op) {
  return ShapeOfXlaOp(op).element_type();
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteValues(
    const xla::XlaOp& op1, const xla::XlaOp& op2) {
  xla::PrimitiveType type1 = TypeOfXlaOp(op1);
  xla::PrimitiveType type2 = TypeOfXlaOp(op2);
  if (type1 == type2) {
    return std::pair<xla::XlaOp, xla::XlaOp>(op1, op2);
  }
  if (type1 == xla::PrimitiveType::F32) {
    return std::pair<xla::XlaOp, xla::XlaOp>(
        op1, xla::ConvertElementType(op2, type1));
  }
  return std::pair<xla::XlaOp, xla::XlaOp>(xla::ConvertElementType(op1, type2),
                                           op2);
}

}  // namespace jit
}  // namespace torch
