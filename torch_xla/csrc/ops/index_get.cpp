#include "torch_xla/csrc/ops/index_get.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& base, const Value& indices,
                           xla::int64 start_dim) {
  auto lower_for_shape_fn =
      [start_dim](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2);
    return CreateIndex(operands[0], operands[1], start_dim);
  };
  return InferOutputShape({base.shape(), indices.shape()}, lower_for_shape_fn);
}

}  // namespace

IndexGet::IndexGet(const ir::Value& base, const ir::Value& indices,
                   xla::int64 start_dim)
    : Node(OpKind(at::aten::index), {base, indices},
           [&]() { return NodeOutputShape(base, indices, start_dim); },
           /*num_outputs=*/1, xla::util::MHash(start_dim)),
      start_dim_(start_dim) {}

std::string IndexGet::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", start_dim=" << start_dim_;
  return ss.str();
}

NodePtr IndexGet::Clone(OpList operands) const {
  return MakeNode<IndexGet>(operands.at(0), operands.at(1), start_dim_);
}

XlaOpVector IndexGet::Lower(LoweringContext* loctx) const {
  xla::XlaOp base = loctx->GetOutputOp(operand(0));
  xla::XlaOp indices = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = CreateIndex(base, indices, start_dim_);
  return ReturnOp(output, loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
