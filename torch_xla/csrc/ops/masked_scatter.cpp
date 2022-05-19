#include "torch_xla/csrc/ops/masked_scatter.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {

MaskedScatter::MaskedScatter(const torch::lazy::Value& input,
                             const torch::lazy::Value& mask,
                             const torch::lazy::Value& source)
    : XlaNode(torch::lazy::OpKind(at::aten::masked_scatter),
              {input, mask, source}, GetXlaShape(input),
              /*num_outputs=*/1) {}

torch::lazy::NodePtr MaskedScatter::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<MaskedScatter>(operands.at(0), operands.at(1),
                                              operands.at(2));
}

XlaOpVector MaskedScatter::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp mask = loctx->GetOutputOp(operand(1));
  xla::XlaOp source = loctx->GetOutputOp(operand(2));
  return ReturnOp(BuildMaskedScatter(input, mask, source), loctx);
}

}  // namespace torch_xla
