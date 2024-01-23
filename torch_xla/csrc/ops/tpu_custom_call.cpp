#include "torch_xla/csrc/ops/tpu_custom_call.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {

TpuCustomCall::TpuCustomCall(const torch::lazy::Value& x, const torch::lazy::Value& y, const std::string& payload)
    : XlaNode(xla_tpu_custom_call, {x, y}, GetXlaShape(x) /*TODO: update it later.*/,
              /*num_outputs=*/1, torch::lazy::MHash(payload))
    , payload_(payload) {}

torch::lazy::NodePtr TpuCustomCall::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<TpuCustomCall>(operands.at(0), operands.at(1), payload_);
}

XlaOpVector TpuCustomCall::Lower(LoweringContext* loctx) const {
  xla::XlaOp x = loctx->GetOutputOp(operand(0));
  xla::XlaOp y = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = BuildTpuCustomCall(x, y, payload_);
  return ReturnOp(output, loctx);
}

std::string TpuCustomCall::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", " << payload_;
  return ss.str();
}

}  // namespace torch_xla
