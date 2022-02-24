#include "torch_xla/csrc/ops/rrelu_with_noise_backward.h"

#include "torch/csrc/lazy/core/ir.h"
#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/scalar.h"

namespace torch_xla {
namespace ir {
namespace ops {

RreluWithNoiseBackward::RreluWithNoiseBackward(
    const Value& grad_output, const Value& input, const Value& noise,
    const at::Scalar& lower, const at::Scalar& upper, bool training)
    : Node(torch::lazy::OpKind(at::aten::rrelu_with_noise_backward),
           {grad_output, input, noise}, input.shape(),
           /*num_outputs=*/1,
           torch::lazy::MHash(ScalarHash(lower), ScalarHash(upper), training)),
      lower_(std::move(lower)),
      upper_(std::move(upper)),
      training_(training) {}

NodePtr RreluWithNoiseBackward::Clone(OpList operands) const {
  return ir::MakeNode<RreluWithNoiseBackward>(operands.at(0), operands.at(1),
                                          operands.at(2), lower_, upper_,
                                          training_);
}

XlaOpVector RreluWithNoiseBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand_with_shape(0));
  xla::XlaOp input = loctx->GetOutputOp(operand_with_shape(1));
  xla::XlaOp noise = loctx->GetOutputOp(operand_with_shape(2));
  return ReturnOp(
      BuildRreluBackward(grad_output, input, noise, lower_, upper_, training_),
      loctx);
}

std::string RreluWithNoiseBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", lower=" << lower_ << ", upper=" << upper_
     << ", training=" << training_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
