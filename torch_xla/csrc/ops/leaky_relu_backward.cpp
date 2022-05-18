#include "torch_xla/csrc/ops/leaky_relu_backward.h"

#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {

LeakyReluBackward::LeakyReluBackward(const torch::lazy::Value& grad_output,
                                     const torch::lazy::Value& input,
                                     double negative_slope)
    : XlaNode(torch::lazy::OpKind(at::aten::leaky_relu_backward),
              {grad_output, input}, GetXlaShape(input),
              /*num_outputs=*/1, torch::lazy::MHash(negative_slope)),
      negative_slope_(negative_slope) {}

torch::lazy::NodePtr LeakyReluBackward::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<LeakyReluBackward>(
      operands.at(0), operands.at(1), negative_slope_);
}

XlaOpVector LeakyReluBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp output =
      BuildLeakyReluBackward(grad_output, input, negative_slope_);
  return ReturnOp(output, loctx);
}

std::string LeakyReluBackward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", negative_slope=" << negative_slope_;
  return ss.str();
}

}  // namespace torch_xla
