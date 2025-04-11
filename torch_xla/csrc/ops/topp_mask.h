#ifndef XLA_TORCH_XLA_CSRC_OPS_TOPP_MASK_H_
#define XLA_TORCH_XLA_CSRC_OPS_TOPP_MASK_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class TopPMask : public XlaNode {
 public:
  TopPMask(const torch::lazy::Value& input, float p, int64_t dim, bool stable);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  float p() const { return p_; };

  int64_t dim() const { return dim_; };

  bool stable() const { return stable_; }

 private:
  float p_;
  int64_t dim_;
  bool stable_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_TOPP_MASK_H_
