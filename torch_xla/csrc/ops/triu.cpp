#include "torch_xla/csrc/ops/triu.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/matrix.h"

namespace torch_xla {

Triu::Triu(const torch::lazy::Value& input, int64_t diagonal)
    : XlaNode(torch::lazy::OpKind(at::aten::triu), {input}, GetXlaShape(input),
              /*num_outputs=*/1, torch::lazy::MHash(diagonal)),
      diagonal_(diagonal) {}

torch::lazy::NodePtr Triu::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Triu>(operands.at(0), diagonal_);
}

XlaOpVector Triu::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildTriu(input, diagonal_);
  return ReturnOp(output, loctx);
}

std::string Triu::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", diagonal=" << diagonal_;
  return ss.str();
}

}  // namespace torch_xla
