#ifndef XLA_TORCH_XLA_CSRC_OPS_CUSTOM_MARK_SHARDING_H_
#define XLA_TORCH_XLA_CSRC_OPS_CUSTOM_MARK_SHARDING_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class CustomMarkSharding : public XlaNode {
 public:
  // Make a custom call to Sharding.
  CustomMarkSharding(const torch::lazy::Value& input,
                     const torch::lazy::Value& sharding);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_CUSTOM_MARK_SHARDING_H_
