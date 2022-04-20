#include "torch_xla/csrc/ops/threshold.h"

#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {

Threshold::Threshold(const Value& input, float threshold, float value)
    : Node(torch::lazy::OpKind(at::aten::threshold), {input}, input.xla_shape(),
           /*num_outputs=*/1, torch::lazy::MHash(threshold, value)),
      threshold_(threshold),
      value_(value) {}

torch::lazy::NodePtr Threshold::Clone(OpList operands) const {
  return ir::MakeNode<Threshold>(operands.at(0), threshold_, value_);
}

XlaOpVector Threshold::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildThreshold(input, input, threshold_, value_);
  return ReturnOp(output, loctx);
}

std::string Threshold::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", threshold=" << threshold_
     << ", value=" << value_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
