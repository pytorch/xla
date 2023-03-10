#ifndef XLA_TORCH_XLA_CSRC_OPS_UNIFORM_H_
#define XLA_TORCH_XLA_CSRC_OPS_UNIFORM_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Uniform : public XlaNode {
 public:
  Uniform(const torch::lazy::Value& from, const torch::lazy::Value& to,
          const torch::lazy::Value& seed, const xla::Shape& rng_shape);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_UNIFORM_H_