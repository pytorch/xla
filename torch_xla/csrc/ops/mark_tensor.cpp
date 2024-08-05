#include "torch_xla/csrc/ops/mark_tensor.h"

#include <torch/csrc/lazy/core/tensor_util.h>

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/shape_helper.h"

namespace torch_xla {

MarkTensor::MarkTensor(const torch::lazy::Value& input, const std::string& info)
    : XlaNode(xla_mark_tensor, {input}, GetXlaShape(input),
              /*num_outputs=*/1, torch::lazy::MHash(info)),
      info_(info) {}

torch::lazy::NodePtr MarkTensor::Clone(torch::lazy::OpList operands) const {
  return torch_xla::MakeNode<MarkTensor>(operands.at(0), info_);
}

XlaOpVector MarkTensor::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::Shape input_shape = ShapeHelper::ShapeOfXlaOp(input);
  static const std::string opname = "xla_mark_tensor";
  xla::XlaOp output =
      xla::CustomCall(input.builder(), opname, {input}, input_shape, info_);
  return ReturnOp(output, loctx);
}

std::string MarkTensor::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", info=" << info_;
  return ss.str();
}

}  // namespace torch_xla
