#include "torch_xla/csrc/ops/recv.h"

#include "third_party/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& token,
                           const xla::Shape& recv_shape, int64_t channel_id) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp tokenOp = operands[0];
    RecvResult result = BuildRecvWithToken(tokenOp, recv_shape, channel_id);
    return xla::Tuple(tokenOp.builder(), {result.result, result.token});
  };
  return InferOutputShape({recv_shape, GetXlaShape(token)}, shape_fn);
}

}  // namespace

Recv::Recv(const torch::lazy::Value& token, const xla::Shape& recv_shape,
           int64_t channel_id)
    : XlaNode(
          xla_recv, {token},
          [&]() { return NodeOutputShape(token, recv_shape, channel_id); },
          /*num_outputs=*/2,
          torch::lazy::MHash(channel_id, recv_shape.ToString())),
      recv_shape_(recv_shape.ToProto()),
      channel_id_(channel_id) {}

torch::lazy::NodePtr Recv::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Recv>(operands.at(0), recv_shape_, channel_id_);
}

XlaOpVector Recv::Lower(LoweringContext* loctx) const {
  xla::XlaOp token = loctx->GetOutputOp(operand(0));
  RecvResult result = BuildRecvWithToken(token, recv_shape_, channel_id_);
  return ReturnOps({result.result, result.token}, loctx);
}

std::string Recv::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", recv shape=" << recv_shape_
     << ", channel_id=" << channel_id_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla