#ifndef XLA_TORCH_XLA_CSRC_OPS_ARG_MAX_H_
#define XLA_TORCH_XLA_CSRC_OPS_ARG_MAX_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class ArgMax : public XlaNode {
 public:
  ArgMax(const torch::lazy::Value& input, int64_t dim, bool keepdim);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t dim() const { return dim_; };

  bool keepdim() const { return keepdim_; }

 private:
  int64_t dim_;
  bool keepdim_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_ARG_MAX_H_