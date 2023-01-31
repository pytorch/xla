#include "torch_xla/csrc/ops/not_supported.h"

#include "xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {

NotSupported::NotSupported(std::string description, xla::Shape shape)
    : XlaNode(xla_not_supported, std::move(shape), /*num_outputs=*/1,
              torch::lazy::MHash(description)),
      description_(std::move(description)) {}

torch::lazy::NodePtr NotSupported::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<NotSupported>(description_, xla_shape());
}

XlaOpVector NotSupported::Lower(LoweringContext* /* loctx */) const {
  XLA_ERROR() << "Node not supported: " << ToString();
}

std::string NotSupported::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", description=" << description_;
  return ss.str();
}

}  // namespace torch_xla
