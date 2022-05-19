#include "torch_xla/csrc/ops/index_put.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {

IndexPut::IndexPut(const torch::lazy::Value& base,
                   const torch::lazy::Value& indices, int64_t start_dim,
                   const torch::lazy::Value& values, bool accumulate)
    : XlaNode(torch::lazy::OpKind(at::aten::index_put), {base, indices, values},
              GetXlaShape(base),
              /*num_outputs=*/1, torch::lazy::MHash(start_dim, accumulate)),
      start_dim_(start_dim),
      accumulate_(accumulate) {}

std::string IndexPut::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", start_dim=" << start_dim_
     << ", accumulate=" << accumulate_;
  return ss.str();
}

torch::lazy::NodePtr IndexPut::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<IndexPut>(
      operands.at(0), operands.at(1), start_dim_, operands.at(2), accumulate_);
}

XlaOpVector IndexPut::Lower(LoweringContext* loctx) const {
  std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)> add_scatter_combiner =
      [](xla::XlaOp x, xla::XlaOp y) -> xla::XlaOp { return x + y; };

  xla::XlaOp base = loctx->GetOutputOp(operand(0));
  xla::XlaOp indices = loctx->GetOutputOp(operand(1));
  xla::XlaOp values = loctx->GetOutputOp(operand(2));
  xla::XlaOp output =
      CreateIndexUpdate(base, indices, start_dim_, values,
                        accumulate_ ? add_scatter_combiner : nullptr);
  return ReturnOp(output, loctx);
}

}  // namespace torch_xla
