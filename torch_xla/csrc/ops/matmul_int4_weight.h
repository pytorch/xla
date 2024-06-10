#ifndef XLA_TORCH_XLA_CSRC_OPS_MATMUL_INT4_WEIGHT
#define XLA_TORCH_XLA_CSRC_OPS_MATMUL_INT4_WEIGHT

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class ReinterpretCast4bit : public XlaNode {
 public:
  ReinterpretCast4bit(const torch::lazy::Value& lhs, const torch::lazy::Value& rhs,
                      const std::vector<int8_t>& int4_weight_values);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  std::vector<int8_t> int4_vals_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_MATMUL_INT4_WEIGHT
