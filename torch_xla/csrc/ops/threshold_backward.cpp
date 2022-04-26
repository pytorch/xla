#include "torch_xla/csrc/ops/threshold_backward.h"

#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {

ThresholdBackward::ThresholdBackward(const XlaValue& grad_output,
                                     const XlaValue& input, float threshold)
    : XlaNode(torch::lazy::OpKind(at::aten::threshold_backward),
           {grad_output, input}, input.xla_shape(), /*num_outputs=*/1,
           torch::lazy::MHash(threshold)),
      threshold_(threshold) {}

torch::lazy::NodePtr ThresholdBackward::Clone(OpList operands) const {
  return ir::MakeNode<ThresholdBackward>(operands.at(0), operands.at(1),
                                         threshold_);
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

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
