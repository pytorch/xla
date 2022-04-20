#include "torch_xla/csrc/ops/not_supported.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

NotSupported::NotSupported(std::string description, xla::Shape shape)
    : Node(xla_not_supported, std::move(shape), /*num_outputs=*/1,
           torch::lazy::MHash(description)),
      description_(std::move(description)) {}

torch::lazy::NodePtr NotSupported::Clone(OpList operands) const {
  return ir::MakeNode<NotSupported>(description_, xla_shape());
}

XlaOpVector NotSupported::Lower(LoweringContext* /* loctx */) const {
  XLA_ERROR() << "Node not supported: " << ToString();
}

std::string NotSupported::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", description=" << description_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
