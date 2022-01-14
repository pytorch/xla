#include "torch_xla/csrc/ops/reduce_scatter.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(AllReduceType reduce_type, const Value input,
                           const Value& token, double scale,
                           int64_t scatter_dim, int64_t shard_count,
                           const std::vector<std::vector<int64_t>>& groups) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp inputOp = operands[0];
    xla::XlaOp tokenOp = operands[1];
    ReduceScatterResult result = BuildReduceScatter(
        reduce_type, inputOp, tokenOp, scale, scatter_dim, shard_count, groups);
    return xla::Tuple(operands[0].builder(), {result.result, result.token});
  };
  return InferOutputShape({input.shape(), token.shape()}, shape_fn);
}

}  // namespace

ReduceScatter::ReduceScatter(AllReduceType reduce_type, const Value& input,
                             const Value& token, double scale,
                             int64_t scatter_dim, int64_t shard_count,
                             std::vector<std::vector<int64_t>> groups)
    : Node(xla_reduce_scatter, {input, token},
           [&]() {
             return NodeOutputShape(reduce_type, input, token, scale,
                                    scatter_dim, shard_count, groups);
           },
           /*num_outputs=*/2,
           torch::lazy::MHash(xla::util::GetEnumValue(reduce_type), scale,
                              scatter_dim, shard_count, groups)),
      reduce_type_(reduce_type),
      scale_(scale),
      scatter_dim_(scatter_dim),
      shard_count_(shard_count),
      groups_(std::move(groups)) {}

NodePtr ReduceScatter::Clone(OpList operands) const {
  return MakeNode<ReduceScatter>(reduce_type_, operands.at(0), operands.at(1),
                                 scale_, scatter_dim_, shard_count_, groups_);
}

XlaOpVector ReduceScatter::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp token = loctx->GetOutputOp(operand(1));
  ReduceScatterResult result = BuildReduceScatter(
      reduce_type_, input, token, scale_, scatter_dim_, shard_count_, groups_);
  return ReturnOps({result.result, result.token}, loctx);
}

std::string ReduceScatter::ToString() const {
  std::stringstream ss;
  ss << Node::ToString()
     << ", reduce_type=" << xla::util::GetEnumValue(reduce_type_)
     << ", scale=" << scale_ << ", scatter_dim=" << scatter_dim_
     << ", shard_count=" << shard_count_ << ", groups=(";
  for (size_t i = 0; i < groups_.size(); ++i) {
    ss << (i == 0 ? "(" : ",(");
    ss << absl::StrJoin(groups_[i], ", ") << ")";
  }
  ss << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
