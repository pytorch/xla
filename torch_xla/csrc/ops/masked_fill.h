#ifndef XLA_TORCH_XLA_CSRC_OPS_MASKED_FILL_H_
#define XLA_TORCH_XLA_CSRC_OPS_MASKED_FILL_H_

#include <c10/core/Scalar.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class MaskedFill : public XlaNode {
 public:
  MaskedFill(const torch::lazy::Value& input, const torch::lazy::Value& mask,
             const at::Scalar& value);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  at::Scalar value() const { return value_; }

 private:
  at::Scalar value_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_MASKED_FILL_H_