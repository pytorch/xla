#include "torch_xla/csrc/ops/index_put.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {

IndexPut::IndexPut(const ir::Value& base, const ir::Value& indices,
                   xla::int64 start_dim, const ir::Value& values,
                   bool accumulate)
    : Node(OpKind(at::aten::index_put), {base, indices, values}, base.shape(),
           /*num_outputs=*/1, xla::util::MHash(start_dim, accumulate)),
      start_dim_(start_dim),
      accumulate_(accumulate) {}

std::string IndexPut::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", start_dim=" << start_dim_
     << ", accumulate=" << accumulate_;
  return ss.str();
}

XlaOpVector IndexPut::Lower(LoweringContext* loctx) const {
  std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp, xla::XlaBuilder*)>
      add_scatter_combiner =
          [](const xla::XlaOp& x, const xla::XlaOp& y,
             xla::XlaBuilder* builder) -> xla::XlaOp { return x + y; };

  xla::XlaOp base = loctx->GetOutputOp(operand(0));
  xla::XlaOp indices = loctx->GetOutputOp(operand(1));
  xla::XlaOp values = loctx->GetOutputOp(operand(2));
  xla::XlaOp output =
      CreateIndexUpdate(base, indices, start_dim_, values,
                        accumulate_ ? add_scatter_combiner : nullptr);
  return ReturnOp(output, loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
