#include "torch_xla/csrc/ops/all_gather.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(
    const Value& input, const Value& token, int64_t dim,
    int64_t shard_count,
    const std::vector<std::vector<int64_t>>& groups) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    AllGatherResult result =
        BuildAllGather(operands[0], operands[1], dim, shard_count, groups);
    return xla::Tuple(operands[0].builder(), {result.result, result.token});
  };
  return InferOutputShape({input.shape(), token.shape()}, shape_fn);
}

}  // namespace

AllGather::AllGather(const Value& input, const Value& token, int64_t dim,
                     int64_t shard_count,
                     std::vector<std::vector<int64_t>> groups)
    : Node(xla_all_gather, {input, token},
           [&]() {
             return NodeOutputShape(input, token, dim, shard_count, groups);
           },
           /*num_outputs=*/2, torch::lazy::MHash(dim, shard_count, groups)),
      dim_(dim),
      shard_count_(shard_count),
      groups_(std::move(groups)) {}

NodePtr AllGather::Clone(OpList operands) const {
  return MakeNode<AllGather>(operands.at(0), operands.at(1), dim_, shard_count_,
                             groups_);
}

XlaOpVector AllGather::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp token = loctx->GetOutputOp(operand(1));
  AllGatherResult result =
      BuildAllGather(input, token, dim_, shard_count_, groups_);
  return ReturnOps({result.result, result.token}, loctx);
}

std::string AllGather::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", shard_count=" << shard_count_
     << ", groups=(";
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
