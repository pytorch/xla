#include "torch_xla/csrc/ops/triu.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/matrix.h"

namespace torch_xla {
namespace ir {
namespace ops {

Triu::Triu(const Value& input, int64_t diagonal)
    : Node(ir::OpKind(at::aten::triu), {input}, input.shape(),
           /*num_outputs=*/1, torch::lazy::MHash(diagonal)),
      diagonal_(diagonal) {}

NodePtr Triu::Clone(OpList operands) const {
  return MakeNode<Triu>(operands.at(0), diagonal_);
}

XlaOpVector Triu::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildTriu(input, diagonal_);
  return ReturnOp(output, loctx);
}

std::string Triu::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", diagonal=" << diagonal_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
