#include "torch_xla/csrc/ops/scatter_add.h"

#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {

ScatterAdd::ScatterAdd(const Value& input, const Value& index, const Value& src,
                       int64_t dim)
    : Node(torch::lazy::OpKind(at::aten::scatter_add), {input, index, src},
           input.xla_shape(),
           /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

torch::lazy::NodePtr ScatterAdd::Clone(OpList operands) const {
  return ir::MakeNode<ScatterAdd>(operands.at(0), operands.at(1),
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
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
