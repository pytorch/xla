#include "torch_xla/csrc/ops/shrink_backward.h"

#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/scalar.h"

namespace torch_xla {
namespace ir {
namespace ops {

ShrinkBackward::ShrinkBackward(OpKind kind, const Value& grad_output,
                               const Value& input, at::Scalar lambda)
    : Node(kind, {grad_output, input}, input.shape(), /*num_outputs=*/1,
           ScalarHash(lambda)),
      lambda_(std::move(lambda)) {}

std::string ShrinkBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", lambda=" << lambda_;
  return ss.str();
}

NodePtr ShrinkBackward::Clone(OpList operands) const {
  return MakeNode<ShrinkBackward>(op(), operands.at(0), operands.at(1),
                                  lambda_);
}

XlaOpVector ShrinkBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildShrinkBackward(grad_output, input, lambda_), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
