#include "torch_xla/csrc/ops/tril.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/matrix.h"

namespace torch_xla {
namespace ir {
namespace ops {

Tril::Tril(const Value& input, int64_t diagonal)
    : Node(ir::OpKind(at::aten::tril), {input}, input.shape(),
           /*num_outputs=*/1, torch::lazy::MHash(diagonal)),
      diagonal_(diagonal) {}

NodePtr Tril::Clone(OpList operands) const {
  return MakeNode<Tril>(operands.at(0), diagonal_);
}

XlaOpVector Tril::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildTril(input, diagonal_);
  return ReturnOp(output, loctx);
}

std::string Tril::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", diagonal=" << diagonal_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
