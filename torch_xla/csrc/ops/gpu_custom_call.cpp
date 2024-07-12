#include "torch_xla/csrc/ops/gpu_custom_call.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {

GpuCustomCall::GpuCustomCall(torch::lazy::OpList inputs,
                             xla::Shape output_shape,
                             const std::string& payload)
    : XlaNode(xla_gpu_custom_call, inputs, std::move(output_shape),
              /*num_outputs=*/output_shape.tuple_shapes_size(),
              torch::lazy::MHash(payload)),
      payload_(payload) {}

torch::lazy::NodePtr GpuCustomCall::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<GpuCustomCall>(operands, xla_shape(), payload_);
}

XlaOpVector GpuCustomCall::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> inputs;
  inputs.reserve(operands().size());
  for (auto& operand : operands()) {
    inputs.push_back(loctx->GetOutputOp(operand));
  }
  auto output = BuildGpuCustomCall(inputs, xla_shape(), payload_);
  return ReturnOps(output, loctx);
}

std::string GpuCustomCall::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", " << payload_;
  return ss.str();
}

}  // namespace torch_xla
