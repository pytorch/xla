#ifndef XLA_TORCH_XLA_CSRC_OPS_DOT_GENERAL_H_
#define XLA_TORCH_XLA_CSRC_OPS_DOT_GENERAL_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class DotGeneral : public XlaNode {
 public:
  DotGeneral(const torch::lazy::Value& lhs, const torch::lazy::Value& rhs,
             const std::vector<std::vector<int>>& dim_vectors,
             std::optional<at::ScalarType> preferred_element_type);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  std::vector<std::vector<int>> dim_vectors_;
  std::optional<at::ScalarType> preferred_element_type_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_DOT_GENERAL_H_
