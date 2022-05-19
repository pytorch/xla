#include "torch_xla/csrc/ops/hardshrink.h"

#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/scalar.h"

namespace torch_xla {

Hardshrink::Hardshrink(const torch::lazy::Value& input,
                       const at::Scalar& lambda)
    : XlaNode(torch::lazy::OpKind(at::aten::hardshrink), {input},
              GetXlaShape(input),
              /*num_outputs=*/1, ScalarHash(lambda)),
      lambda_(std::move(lambda)) {}

std::string Hardshrink::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", lambda=" << lambda_;
  return ss.str();
}

torch::lazy::NodePtr Hardshrink::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Hardshrink>(operands.at(0), lambda_);
}

XlaOpVector Hardshrink::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildHardshrink(input, lambda_), loctx);
}

}  // namespace torch_xla
