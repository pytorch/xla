#include "torch_xla/csrc/ops/shrink_backward.h"

#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/scalar.h"

namespace torch_xla {

ShrinkBackward::ShrinkBackward(torch::lazy::OpKind kind,
                               const torch::lazy::Value& grad_output,
                               const torch::lazy::Value& input,
                               const at::Scalar& lambda)
    : XlaNode(kind, {grad_output, input}, GetXlaShape(input), /*num_outputs=*/1,
              ScalarHash(lambda)),
      lambda_(std::move(lambda)) {}

std::string ShrinkBackward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", lambda=" << lambda_;
  return ss.str();
}

torch::lazy::NodePtr ShrinkBackward::Clone(OpList operands) const {
  return torch::lazy::MakeNode<ShrinkBackward>(op(), operands.at(0),
                                               operands.at(1), lambda_);
}

XlaOpVector ShrinkBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildShrinkBackward(grad_output, input, lambda_), loctx);
}

}  // namespace torch_xla
