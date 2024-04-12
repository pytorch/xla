#ifndef XLA_TORCH_XLA_CSRC_OPS_CUSTOM_SHARDING_H_
#define XLA_TORCH_XLA_CSRC_OPS_CUSTOM_SHARDING_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class CustomSharding : public XlaNode {
 public:
  enum class Type {
    kSharding,
    kSPMDFullToShardShape,
    kSPMDShardToFullShape,
  };

  // Make a custom call to Sharding.
  CustomSharding(const torch::lazy::Value& input,
                 const xla::Shape& output_shape, const Type& type);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  Type type;
  xla::Shape output_shape;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_CUSTOM_SHARDING_H_
