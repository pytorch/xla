#ifndef XLA_TORCH_XLA_CSRC_OPS_EINSUM_BACKWARD_H_
#define XLA_TORCH_XLA_CSRC_OPS_EINSUM_BACKWARD_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class EinsumBackward : public XlaNode {
 public:
  EinsumBackward(const torch::lazy::Value& grad_output,
                 const torch::lazy::OpList& inputs, const std::string equation);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::string& equation() const { return equation_; }

 private:
  const std::string equation_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_EINSUM_BACKWARD_H_