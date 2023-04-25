#include "torch_xla/csrc/ops/reduce_scatter.h"

#include <torch/csrc/lazy/core/util.h>

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(AllReduceType reduce_type,
                           const torch::lazy::Value input,
                           const torch::lazy::Value& token, double scale,
                           int64_t scatter_dim, int64_t shard_count,
                           const std::vector<std::vector<int64_t>>& groups,
                           bool pin_layout) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp inputOp = operands[0];
    xla::XlaOp tokenOp = operands[1];
    ReduceScatterResult result =
        BuildReduceScatter(reduce_type, inputOp, tokenOp, scale, scatter_dim,
                           shard_count, groups, pin_layout);
    return xla::Tuple(operands[0].builder(), {result.result, result.token});
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(token)}, shape_fn);
}

}  // namespace

ReduceScatter::ReduceScatter(AllReduceType reduce_type,
                             const torch::lazy::Value& input,
                             const torch::lazy::Value& token, double scale,
                             int64_t scatter_dim, int64_t shard_count,
                             std::vector<std::vector<int64_t>> groups,
                             bool pin_layout)
    : XlaNode(xla_reduce_scatter, {input, token},
              [&]() {
                return NodeOutputShape(reduce_type, input, token, scale,
                                       scatter_dim, shard_count, groups,
                                       pin_layout);
              },
              /*num_outputs=*/2,
              torch::lazy::MHash(torch::lazy::GetEnumValue(reduce_type), scale,
                                 scatter_dim, shard_count, groups, pin_layout)),
      reduce_type_(reduce_type),
      scale_(scale),
      scatter_dim_(scatter_dim),
      shard_count_(shard_count),
      groups_(std::move(groups)),
      pin_layout_(pin_layout) {}

torch::lazy::NodePtr ReduceScatter::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<ReduceScatter>(
      reduce_type_, operands.at(0), operands.at(1), scale_, scatter_dim_,
      shard_count_, groups_, pin_layout_);
}

XlaOpVector ReduceScatter::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp token = loctx->GetOutputOp(operand(1));
  ReduceScatterResult result =
      BuildReduceScatter(reduce_type_, input, token, scale_, scatter_dim_,
                         shard_count_, groups_, pin_layout_);
  return ReturnOps({result.result, result.token}, loctx);
}

std::string ReduceScatter::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString()
     << ", reduce_type=" << torch::lazy::GetEnumValue(reduce_type_)
     << ", scale=" << scale_ << ", scatter_dim=" << scatter_dim_
     << ", shard_count=" << shard_count_ << ", pin_layout=" << pin_layout_
     << ", groups=(";
  for (size_t i = 0; i < groups_.size(); ++i) {
    ss << (i == 0 ? "(" : ",(");
    ss << absl::StrJoin(groups_[i], ", ") << ")";
  }
  ss << ")";
  return ss.str();
}

}  // namespace torch_xla
