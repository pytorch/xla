#include "torch_xla/csrc/ops/scatter.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {

Scatter::Scatter(const Value& input, const Value& index, const Value& src,
                 xla::int64 dim)
    : Node(ir::OpKind(at::aten::scatter), {input, index, src}, input.shape(),
           /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

NodePtr Scatter::Clone(OpList operands) const {
  return MakeNode<Scatter>(operands.at(0), operands.at(1), operands.at(2),
                           dim_);
}

XlaOpVector Scatter::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  xla::XlaOp src = loctx->GetOutputOp(operand(2));

  ScatterOptions options(/*combiner=*/nullptr);
  return ReturnOp(
      CreateScatter(loctx->device(), input, index, src, dim_, options), loctx);
}

std::string Scatter::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
