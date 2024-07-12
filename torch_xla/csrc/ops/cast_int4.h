#ifndef XLA_TORCH_XLA_CSRC_OPS_CAST_INT4
#define XLA_TORCH_XLA_CSRC_OPS_CAST_INT4

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class CastInt4 : public XlaNode {
 public:
  CastInt4(const torch::lazy::Value& weight,
           const std::vector<int>& int4_weight_values);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  std::vector<int> int4_vals_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_CAST_INT4
