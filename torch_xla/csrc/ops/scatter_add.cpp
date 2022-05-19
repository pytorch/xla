#include "torch_xla/csrc/ops/scatter_add.h"

#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {

ScatterAdd::ScatterAdd(const torch::lazy::Value& input,
                       const torch::lazy::Value& index,
                       const torch::lazy::Value& src, int64_t dim)
    : XlaNode(torch::lazy::OpKind(at::aten::scatter_add), {input, index, src},
              GetXlaShape(input),
              /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

torch::lazy::NodePtr ScatterAdd::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<ScatterAdd>(operands.at(0), operands.at(1),
                                           operands.at(2), dim_);
}

XlaOpVector ScatterAdd::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  xla::XlaOp src = loctx->GetOutputOp(operand(2));
  ScatterOptions options(NumericAddCombiner());
  return ReturnOp(
      CreateScatter(loctx->device(), input, index, src, dim_, options), loctx);
}

std::string ScatterAdd::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace torch_xla
