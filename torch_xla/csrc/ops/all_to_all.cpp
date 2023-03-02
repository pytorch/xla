#include "torch_xla/csrc/ops/all_to_all.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           const torch::lazy::Value& token,
                           int64_t split_dimension, int64_t concat_dimension,
                           int64_t split_count,
                           const std::vector<std::vector<int64_t>>& groups,
                           bool pin_layout) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    AllToAllResult result =
        BuildAllToAll(operands[0], operands[1], split_dimension,
                      concat_dimension, split_count, groups, pin_layout);
    return xla::Tuple(operands[0].builder(), {result.result, result.token});
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(token)}, shape_fn);
}

}  // namespace

AllToAll::AllToAll(const torch::lazy::Value& input,
                   const torch::lazy::Value& token, int64_t split_dimension,
                   int64_t concat_dimension, int64_t split_count,
                   std::vector<std::vector<int64_t>> groups, bool pin_layout)
    : XlaNode(
          xla_all_to_all, {input, token},
          [&]() {
            return NodeOutputShape(input, token, split_dimension,
                                   concat_dimension, split_count, groups,
                                   pin_layout);
          },
          /*num_outputs=*/2,
          torch::lazy::MHash(split_dimension, concat_dimension, split_count,
                             groups, pin_layout)),
      split_dimension_(split_dimension),
      concat_dimension_(concat_dimension),
      split_count_(split_count),
      groups_(std::move(groups)),
      pin_layout_(pin_layout) {}

torch::lazy::NodePtr AllToAll::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<AllToAll>(operands.at(0), operands.at(1),
                                         split_dimension_, concat_dimension_,
                                         split_count_, groups_, pin_layout_);
}

XlaOpVector AllToAll::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp token = loctx->GetOutputOp(operand(1));
  AllToAllResult result =
      BuildAllToAll(input, token, split_dimension_, concat_dimension_,
                    split_count_, groups_, pin_layout_);
  return ReturnOps({result.result, result.token}, loctx);
}

std::string AllToAll::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", split_dimension=" << split_dimension_
     << ", concat_dimension=" << concat_dimension_
     << ", split_count=" << split_count_ << ", pin_layout=" << pin_layout_
     << ", groups=(";
  for (size_t i = 0; i < groups_.size(); ++i) {
    ss << (i == 0 ? "(" : ",(");
    ss << absl::StrJoin(groups_[i], ", ") << ")";
  }
  ss << ")";
  return ss.str();
}

}  // namespace torch_xla
