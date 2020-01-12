#include "torch_xla/csrc/ops/masked_scatter.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {

MaskedScatter::MaskedScatter(const Value& input, const Value& mask,
                             const Value& source)
    : Node(ir::OpKind(at::aten::masked_scatter), {input, mask, source},
           input.shape(),
           /*num_outputs=*/1) {}

NodePtr MaskedScatter::Clone(OpList operands) const {
  return MakeNode<MaskedScatter>(operands.at(0), operands.at(1),
                                 operands.at(2));
}

XlaOpVector MaskedScatter::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp mask = loctx->GetOutputOp(operand(1));
  xla::XlaOp source = loctx->GetOutputOp(operand(2));
  return ReturnOp(BuildMaskedScatter(input, mask, source), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
