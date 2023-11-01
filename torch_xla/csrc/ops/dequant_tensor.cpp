#include "torch_xla/csrc/ops/dequant_tensor.h"

#include <torch/csrc/lazy/core/tensor_util.h>

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/quant_util.h"
#include "torch_xla/csrc/shape_helper.h"

namespace torch_xla {

DequantizeTensor::DequantizeTensor(const torch::lazy::Value& input,
                                   const std::vector<float>& scale,
                                   const std::vector<float>& zero_point,
                                   int quant_min, int quant_max,
                                   const std::string& dtype, int axis)
    : XlaNode(
          xla_quantize_per_tensor, {input},
          GetXlaShape(input) /* fix when quant type is added to HLO */,
          /*num_outputs=*/1,
          torch::lazy::MHash(scale, zero_point, quant_min, quant_max, dtype)),
      quant_min_(quant_min),
      quant_max_(quant_max),
      axis_(axis),
      dtype_(dtype),
      scale_(scale),
      zero_point_(zero_point) {}

torch::lazy::NodePtr DequantizeTensor::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<DequantizeTensor>(operands.at(0), scale_,
                                                 zero_point_, quant_min_,
                                                 quant_max_, dtype_, axis_);
}

XlaOpVector DequantizeTensor::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::Shape input_shape = ShapeHelper::ShapeOfXlaOp(input);

  static const std::string opname = "stablehlo.uniform_dequantize";
  auto qparams = QuantParams(scale_, zero_point_, quant_min_, quant_max_, axis_,
                             dtype_, input_shape.element_type());

  xla::XlaOp output = xla::CustomCall(
      input.builder(), opname, {input}, input_shape,
      qparams.SerializeToAttrDictStr(),
      /*has_side_effect=*/false,
      /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
      /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
      /*api_version=*/xla::CustomCallApiVersion::API_VERSION_TYPED_FFI);
  return ReturnOp(output, loctx);
}

std::string DequantizeTensor::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", quant_min=" << quant_min_
     << ", quant_max=" << quant_max_ << ", dtype=" << dtype_;
  return ss.str();
}

}  // namespace torch_xla
