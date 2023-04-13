#include "torch_xla/csrc/ops/flip.h"

#include "xla/client/xla_builder.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {

Flip::Flip(const torch::lazy::Value& input, std::vector<int64_t> dims)
    : XlaNode(torch::lazy::OpKind(at::aten::flip), {input}, GetXlaShape(input),
              /*num_outputs=*/1, torch::lazy::MHash(dims)),
      dims_(std::move(dims)) {}

torch::lazy::NodePtr Flip::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Flip>(operands.at(0), dims_);
}

XlaOpVector Flip::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = xla::Rev(input, dims_);
  return ReturnOp(output, loctx);
}

std::string Flip::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dims=(" << absl::StrJoin(dims_, ", ") << ")";
  return ss.str();
}

}  // namespace torch_xla
