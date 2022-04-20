#include "torch_xla/csrc/ops/put.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {

Put::Put(const Value& input, const Value& index, const Value& source,
         bool accumulate)
    : Node(torch::lazy::OpKind(at::aten::put), {input, index, source},
           input.xla_shape(),
           /*num_outputs=*/1, torch::lazy::MHash(accumulate)),
      accumulate_(accumulate) {}

torch::lazy::NodePtr Put::Clone(OpList operands) const {
  return ir::MakeNode<Put>(operands.at(0), operands.at(1), operands.at(2),
                           accumulate_);
}

XlaOpVector Put::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  xla::XlaOp source = loctx->GetOutputOp(operand(2));
  return ReturnOp(CreatePut(loctx->device(), input, index, source, accumulate_),
                  loctx);
}

std::string Put::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", accumulate=" << accumulate_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
