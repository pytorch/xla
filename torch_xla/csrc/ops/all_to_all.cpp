#include "torch_xla/csrc/ops/all_to_all.h"

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

xla::Shape NodeOutputShape(const Value& input, const Value& token,
                           xla::int64 split_dimension,
                           xla::int64 concat_dimension, xla::int64 split_count,
                           const std::vector<std::vector<xla::int64>>& groups) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    AllToAllResult result =
        BuildAllToAll(operands[0], operands[1], split_dimension,
                      concat_dimension, split_count, groups);
    return xla::Tuple(operands[0].builder(), {result.result, result.token});
  };
  return InferOutputShape({input.shape(), token.shape()}, shape_fn);
}

}  // namespace

AllToAll::AllToAll(const Value& input, const Value& token,
                   xla::int64 split_dimension, xla::int64 concat_dimension,
                   xla::int64 split_count,
                   std::vector<std::vector<xla::int64>> groups)
    : Node(xla_all_to_all, {input, token},
           [&]() {
             return NodeOutputShape(input, token, split_dimension,
                                    concat_dimension, split_count, groups);
           },
           /*num_outputs=*/2,
           xla::util::MHash(split_dimension, concat_dimension, split_count,
                            groups)),
      split_dimension_(split_dimension),
      concat_dimension_(concat_dimension),
      split_count_(split_count),
      groups_(std::move(groups)) {}

NodePtr AllToAll::Clone(OpList operands) const {
  return MakeNode<AllToAll>(operands.at(0), operands.at(1), split_dimension_,
                            concat_dimension_, split_count_, groups_);
}

XlaOpVector AllToAll::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp token = loctx->GetOutputOp(operand(1));
  AllToAllResult result = BuildAllToAll(
      input, token, split_dimension_, concat_dimension_, split_count_, groups_);
  return ReturnOps({result.result, result.token}, loctx);
}

std::string AllToAll::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", split_dimension=" << split_dimension_
     << ", concat_dimension=" << concat_dimension_
     << ", split_count=" << split_count_ << ", groups=(";
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
