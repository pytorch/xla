#include "torch_xla/csrc/ops/custom_sharding.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {

CustomSharding::CustomSharding(const torch::lazy::Value& input)
    : XlaNode(xla_custom_sharding, {input}, GetXlaShape(input),
              /*num_outputs=*/1, torch::lazy::MHash(std::string("Sharding"))) {}

torch::lazy::NodePtr CustomSharding::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<CustomSharding>(operands.at(0));
}

XlaOpVector CustomSharding::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildCustomSharding(input);
  return ReturnOp(output, loctx);
}

std::string CustomSharding::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", Sharding";
  return ss.str();
}

}  // namespace torch_xla
