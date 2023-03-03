#include "torch_xla/csrc/ops/replication_pad_backward.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& grad_output,
                           const torch::lazy::Value& input,
                           absl::Span<const int64_t> padding) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildReplicationPadBackward(operands[0], operands[1], padding);
  };
  return InferOutputShape({GetXlaShape(grad_output), GetXlaShape(input)},
                          lower_for_shape_fn);
}

}  // namespace

ReplicationPadBackward::ReplicationPadBackward(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
    std::vector<int64_t> padding)
    : XlaNode(xla_replication_pad_backward, {grad_output, input},
              [&]() { return NodeOutputShape(grad_output, input, padding); },
              /*num_outputs=*/1, torch::lazy::MHash(padding)),
      padding_(std::move(padding)) {}

torch::lazy::NodePtr ReplicationPadBackward::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<ReplicationPadBackward>(
      operands.at(0), operands.at(1), padding_);
}

XlaOpVector ReplicationPadBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = BuildReplicationPadBackward(grad_output, input, padding_);
  return ReturnOp(output, loctx);
}

std::string ReplicationPadBackward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", padding=(" << absl::StrJoin(padding_, ", ")
     << ")";
  return ss.str();
}

}  // namespace torch_xla
