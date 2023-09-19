#include "torch_xla/csrc/ops/quant_per_tensor.h"

#include <torch/csrc/lazy/core/tensor_util.h>

#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "xla/primitive_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           xla::PrimitiveType type) {
  xla::Shape shape = GetXlaShape(input);
  shape.set_element_type(type);
  return shape;
}

}  // namespace

QuantizePerTensor::QuantizePerTensor(const torch::lazy::Value& input,
                                     const std::vector<float>& scale,
                                     const std::vector<float>& zero_point,
                                     int quant_min, int quant_max,
                                     const std::string& dtype)
    : XlaNode(
          xla_quantize_per_tensor, {input},
          GetXlaShape(input) /* fix when quant type is added to HLO */,
          /*num_outputs=*/1,
          // torch::lazy::MHash(quant_min, quant_max, static_cast<int>(dtype))),
          torch::lazy::MHash(quant_min, quant_max, dtype)),
      quant_min_(quant_min),
      quant_max_(quant_max),
      dtype_(dtype),
      scale_(scale),
      zero_point_(zero_point) {}

torch::lazy::NodePtr QuantizePerTensor::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<QuantizePerTensor>(
      operands.at(0), scale_, zero_point_, quant_min_, quant_max_, dtype_);
}

// XlaOpVector QuantizePerTensor::Lower(LoweringContext* loctx) const {
//   // torch.clamp(torch.round(input * inv_scale) + zero_point, quant_min,
//   quant_max).to(dtype) xla::XlaOp input = loctx->GetOutputOp(operand(0));
//   xla::XlaOp scale = loctx->GetOutputOp(operand(1));
//   xla::XlaOp zero_point = loctx->GetOutputOp(operand(2));

//   xla::XlaOp output = xla::Add(xla::Div(input, scale), zero_point);
//   return ReturnOp(output, loctx);
// }

static inline std::string MaybeAppendDecimalForInteger(float v) {
  std::stringstream ss;
  if (static_cast<int>(v) == v) {
    ss << std::fixed << std::setprecision(2);
  }
  ss << v;
  return ss.str();
}

static std::string SeralizeFloatVector(const std::vector<float>& v, bool append_decimal=false) {
  std::stringstream ss;
  ss << '[';
  for (size_t i = 0; i < v.size(); ++i) {
    if (i != 0) {
      ss << ',';
    }
    if (append_decimal) {
      ss << MaybeAppendDecimalForInteger(v.at(i));
    } else{
      ss << v.at(i);
    }
  }
  ss << ']';
  return ss.str();
}

// TODO: add more types
static std::unordered_map<std::string, std::string> _type_str_map {
  {"torch.int8", "si8"}
};

XlaOpVector QuantizePerTensor::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::Shape input_shape = ShapeHelper::ShapeOfXlaOp(input);

  const std::string opname = "stablehlo.uniform_quantize";
  std::stringstream ss;
  ss << "{";
  // ss << "quantization_dimension=0,";
  ss << "scale=" << SeralizeFloatVector(scale_, true) << ',';
  // ss << "scale=" << "[2]" << ',';
  ss << "zero_point=" << SeralizeFloatVector(zero_point_) << ',';
  ss << "storage_type=" << _type_str_map.at(dtype_) << ',';
  ss << "expressed_type=" << "f32" << ','; /* should equal to the input scalar type*/
  ss << "storage_min=" << quant_min_ << ',';
  ss << "storage_max=" << quant_max_;
  ss << '}';
  // ss << "{quantization_dimension=0, scale=[1.0,2.0], zero_point = [1,1], storage_type=si8, expressed_type=f32, storage_min=-128, storage_max=127}";

  xla::XlaOp output =
      xla::CustomCall(input.builder(), opname, {input}, input_shape, ss.str(),
/*has_side_effect=*/false,
/*output_operand_aliasing=*/{}, /*literal=*/nullptr,
/*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
/*api_version=*/xla::CustomCallApiVersion::API_VERSION_TYPED_FFI);
  return ReturnOp(output, loctx);
}

std::string QuantizePerTensor::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", quant_min=" << quant_min_
     << ", quant_max=" << quant_max_ << ", stype=" << dtype_;
  return ss.str();
}

}  // namespace torch_xla
