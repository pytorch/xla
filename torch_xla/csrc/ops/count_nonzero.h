#ifndef XLA_TORCH_XLA_CSRC_OPS_COUNT_NONZERO_H_
#define XLA_TORCH_XLA_CSRC_OPS_COUNT_NONZERO_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class CountNonzero : public XlaNode {
 public:
  CountNonzero(const torch::lazy::Value& input, std::vector<int64_t> dims);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  c10::optional<std::vector<int64_t>> dims() const { return dims_; }

 private:
  std::vector<int64_t> dims_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_COUNT_NONZERO_H_
