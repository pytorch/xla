#include "helpers.h"

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"

namespace torch {
namespace jit {
namespace {

bool ShouldUseBF16() {
  int use_fp16 = xla::sys_util::GetEnvInt("XLA_USE_BF16", 0);
  if (use_fp16 != 0) {
    TF_LOG(INFO) << "Using BF16 data type for floating point values";
  }
  return use_fp16 != 0;
}

}  // namespace

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

xla::XlaComputation XlaHelpers::CreateAddComputation(xla::PrimitiveType type) {
  xla::XlaBuilder reduction_builder("xla_add_computation");
  const auto x = xla::Parameter(&reduction_builder, 0,
                                xla::ShapeUtil::MakeShape(type, {}), "x");
  const auto y = xla::Parameter(&reduction_builder, 1,
                                xla::ShapeUtil::MakeShape(type, {}), "y");
  Add(x, y);
  return reduction_builder.Build().ConsumeValueOrDie();
}

xla::PrimitiveType XlaHelpers::MakeXlaPrimitiveType(
    const at::ScalarType scalar_type) {
  // When PyTorch will support native BF16 type, the global configuration can be
  // replaced (or augmented) with the proper mapping.
  switch (scalar_type) {
    case at::ScalarType::Float:
      return UseBF16() ? xla::PrimitiveType::BF16 : xla::PrimitiveType::F32;
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
  return op.builder()->GetShape(op).ConsumeValueOrDie();
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
  xla::int64 size1 = xla::ShapeUtil::ByteSizeOfPrimitiveType(type1);
  xla::int64 size2 = xla::ShapeUtil::ByteSizeOfPrimitiveType(type2);
  if (xla::primitive_util::IsFloatingPointType(type1)) {
    if (!xla::primitive_util::IsFloatingPointType(type2) || size1 >= size2) {
      return std::pair<xla::XlaOp, xla::XlaOp>(
          op1, xla::ConvertElementType(op2, type1));
    }
    return std::pair<xla::XlaOp, xla::XlaOp>(
        xla::ConvertElementType(op1, type2), op2);
  }
  if (xla::primitive_util::IsFloatingPointType(type2) || size2 >= size1) {
    return std::pair<xla::XlaOp, xla::XlaOp>(
        xla::ConvertElementType(op1, type2), op2);
  }
  return std::pair<xla::XlaOp, xla::XlaOp>(op1,
                                           xla::ConvertElementType(op2, type1));
}

bool XlaHelpers::UseBF16() {
  static bool use_fp16 = ShouldUseBF16();
  return use_fp16;
}

}  // namespace jit
}  // namespace torch
