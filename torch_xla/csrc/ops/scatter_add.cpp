#include "torch_xla/csrc/ops/scatter_add.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {

ScatterAdd::ScatterAdd(const Value& input, const Value& index, const Value& src,
                       xla::int64 dim)
    : Node(ir::OpKind(at::aten::scatter_add), {input, index, src},
           input.shape(),
           /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

NodePtr ScatterAdd::Clone(OpList operands) const {
  return MakeNode<ScatterAdd>(operands.at(0), operands.at(1), operands.at(2),
                              dim_);
}

XlaOpVector ScatterAdd::Lower(LoweringContext* loctx) const {
  auto add_scatter_combiner = [](const xla::XlaOp& x,
                                 const xla::XlaOp& y) -> xla::XlaOp {
    return x + y;
  };

  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  xla::XlaOp src = loctx->GetOutputOp(operand(2));
  return ReturnOp(CreateScatter(input, index, src, dim_, add_scatter_combiner),
                  loctx);
}

std::string ScatterAdd::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
