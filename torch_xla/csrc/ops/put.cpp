#include "torch_xla/csrc/ops/put.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {

Put::Put(const torch::lazy::Value& input, const torch::lazy::Value& index,
         const torch::lazy::Value& source, bool accumulate)
    : XlaNode(torch::lazy::OpKind(at::aten::put), {input, index, source},
              GetXlaShape(input),
              /*num_outputs=*/1, torch::lazy::MHash(accumulate)),
      accumulate_(accumulate) {}

torch::lazy::NodePtr Put::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Put>(operands.at(0), operands.at(1),
                                    operands.at(2), accumulate_);
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
  ss << XlaNode::ToString() << ", accumulate=" << accumulate_;
  return ss.str();
}

}  // namespace torch_xla
