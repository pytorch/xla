#include "torch_xla/csrc/ops/send.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           const torch::lazy::Value& token,
                           int64_t channel_id) {
  auto shape_fn =
      [channel_id](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp inputOp = operands[0];
    xla::XlaOp tokenOp = operands[1];
    SendResult result = BuildSendWithToken(inputOp, tokenOp, channel_id);
    return xla::Tuple(tokenOp.builder(),
                      {result.input_as_result, result.token});
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(token)}, shape_fn);
}

}  // namespace

Send::Send(const torch::lazy::Value& input, const torch::lazy::Value& token,
           int64_t channel_id)
    : XlaNode(xla_send, {input, token},
              [&]() { return NodeOutputShape(input, token, channel_id); },
              /*num_outputs=*/2, torch::lazy::MHash(channel_id)),
      channel_id_(channel_id) {}

torch::lazy::NodePtr Send::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Send>(operands.at(0), operands.at(1),
                                     channel_id_);
}

XlaOpVector Send::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp token = loctx->GetOutputOp(operand(1));
  SendResult result = BuildSendWithToken(input, token, channel_id_);
  return ReturnOps({result.input_as_result, result.token}, loctx);
}

std::string Send::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", channel_id=" << channel_id_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla