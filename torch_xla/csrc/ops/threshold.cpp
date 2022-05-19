#include "torch_xla/csrc/ops/threshold.h"

#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {

Threshold::Threshold(const torch::lazy::Value& input, float threshold,
                     float value)
    : XlaNode(torch::lazy::OpKind(at::aten::threshold), {input},
              GetXlaShape(input),
              /*num_outputs=*/1, torch::lazy::MHash(threshold, value)),
      threshold_(threshold),
      value_(value) {}

torch::lazy::NodePtr Threshold::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Threshold>(operands.at(0), threshold_, value_);
}

XlaOpVector Threshold::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildThreshold(input, input, threshold_, value_);
  return ReturnOp(output, loctx);
}

std::string Threshold::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", threshold=" << threshold_
     << ", value=" << value_;
  return ss.str();
}

}  // namespace torch_xla
