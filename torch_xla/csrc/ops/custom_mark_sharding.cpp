#include "torch_xla/csrc/ops/custom_mark_sharding.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {

CustomMarkSharding::CustomMarkSharding(const torch::lazy::Value& input)
    : XlaNode(xla_custom_mark_sharding, {input}, GetXlaShape(input),
              /*num_outputs=*/1, torch::lazy::MHash(std::string("MarkSharding"))) {}

torch::lazy::NodePtr CustomMarkSharding::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<CustomMarkSharding>(operands.at(0));
}

XlaOpVector CustomMarkSharding::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildCustomMarkSharding(input);
  return ReturnOp(output, loctx);
}

std::string CustomMarkSharding::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", MarkSharding";
  return ss.str();
}

}  // namespace torch_xla
