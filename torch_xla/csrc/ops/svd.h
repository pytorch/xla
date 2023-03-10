#ifndef XLA_TORCH_XLA_CSRC_OPS_SVD_H_
#define XLA_TORCH_XLA_CSRC_OPS_SVD_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class SVD : public XlaNode {
 public:
  SVD(const torch::lazy::Value& input, bool some, bool compute_uv);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  bool some() const { return some_; }

  bool compute_uv() const { return compute_uv_; }

 private:
  bool some_;
  bool compute_uv_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_SVD_H_