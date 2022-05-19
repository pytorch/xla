#include "torch_xla/csrc/ops/threshold_backward.h"

#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {

ThresholdBackward::ThresholdBackward(const torch::lazy::Value& grad_output,
                                     const torch::lazy::Value& input,
                                     float threshold)
    : XlaNode(torch::lazy::OpKind(at::aten::threshold_backward),
              {grad_output, input}, GetXlaShape(input), /*num_outputs=*/1,
              torch::lazy::MHash(threshold)),
      threshold_(threshold) {}

torch::lazy::NodePtr ThresholdBackward::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<ThresholdBackward>(operands.at(0),
                                                  operands.at(1), threshold_);
}

XlaOpVector ThresholdBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = BuildThreshold(input, grad_output, threshold_, 0);
  return ReturnOp(output, loctx);
}

std::string ThresholdBackward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", threshold=" << threshold_;
  return ss.str();
}

}  // namespace torch_xla
