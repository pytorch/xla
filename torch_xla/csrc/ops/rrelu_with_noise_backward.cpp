#include "torch_xla/csrc/ops/rrelu_with_noise_backward.h"

#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/scalar.h"

namespace torch_xla {

RreluWithNoiseBackward::RreluWithNoiseBackward(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
    const torch::lazy::Value& noise, const at::Scalar& lower,
    const at::Scalar& upper, bool training)
    : XlaNode(
          torch::lazy::OpKind(at::aten::rrelu_with_noise_backward),
          {grad_output, input, noise}, GetXlaShape(input),
          /*num_outputs=*/1,
          torch::lazy::MHash(ScalarHash(lower), ScalarHash(upper), training)),
      lower_(std::move(lower)),
      upper_(std::move(upper)),
      training_(training) {}

torch::lazy::NodePtr RreluWithNoiseBackward::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<RreluWithNoiseBackward>(
      operands.at(0), operands.at(1), operands.at(2), lower_, upper_,
      training_);
}

XlaOpVector RreluWithNoiseBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp noise = loctx->GetOutputOp(operand(2));
  return ReturnOp(
      BuildRreluBackward(grad_output, input, noise, lower_, upper_, training_),
      loctx);
}

std::string RreluWithNoiseBackward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", lower=" << lower_ << ", upper=" << upper_
     << ", training=" << training_;
  return ss.str();
}

}  // namespace torch_xla
