#include "torch_xla/csrc/ops/tril.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/matrix.h"

namespace torch_xla {

Tril::Tril(const XlaValue& input, int64_t diagonal)
    : XlaNode(torch::lazy::OpKind(at::aten::tril), {input}, input.xla_shape(),
              /*num_outputs=*/1, torch::lazy::MHash(diagonal)),
      diagonal_(diagonal) {}

torch::lazy::NodePtr Tril::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Tril>(operands.at(0), diagonal_);
}

XlaOpVector Tril::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildTril(input, diagonal_);
  return ReturnOp(output, loctx);
}

std::string Tril::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", diagonal=" << diagonal_;
  return ss.str();
}

} // namespace torch_xla
