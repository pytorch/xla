#include "torch_xla/csrc/ops/tpu_custom_call.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {

TpuCustomCall::TpuCustomCall(torch::lazy::OpList inputs,
                             xla::Shape output_shape,
                             const std::string& payload)
    : XlaNode(xla_tpu_custom_call, inputs, output_shape,
              /*num_outputs=*/output_shape.tuple_shapes_size(),
              torch::lazy::MHash(payload)),
      payload_(payload) {}

torch::lazy::NodePtr TpuCustomCall::Clone(torch::lazy::OpList operands) const {
  return torch_xla::MakeNode<TpuCustomCall>(operands, xla_shape(), payload_);
}

XlaOpVector TpuCustomCall::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> inputs;
  inputs.reserve(operands().size());
  for (auto& operand : operands()) {
    inputs.push_back(loctx->GetOutputOp(operand));
  }
  auto output = BuildTpuCustomCall(inputs, xla_shape(), payload_);
  return ReturnOps(output, loctx);
}

std::string TpuCustomCall::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", " << payload_;
  return ss.str();
}

}  // namespace torch_xla
