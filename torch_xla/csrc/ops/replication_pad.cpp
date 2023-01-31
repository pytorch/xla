#include "torch_xla/csrc/ops/replication_pad.h"

#include "xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           absl::Span<const int64_t> padding) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildReplicationPad(operands[0], padding);
  };
  return InferOutputShape({GetXlaShape(input)}, shape_fn);
}

}  // namespace

ReplicationPad::ReplicationPad(const torch::lazy::Value& input,
                               std::vector<int64_t> padding)
    : XlaNode(xla_replication_pad, {input},
              [&]() { return NodeOutputShape(input, padding); },
              /*num_outputs=*/1, torch::lazy::MHash(padding)),
      padding_(std::move(padding)) {}

torch::lazy::NodePtr ReplicationPad::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<ReplicationPad>(operands.at(0), padding_);
}

XlaOpVector ReplicationPad::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildReplicationPad(input, padding_);
  return ReturnOp(output, loctx);
}

std::string ReplicationPad::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", padding=(" << absl::StrJoin(padding_, ", ")
     << ")";
  return ss.str();
}

}  // namespace torch_xla
