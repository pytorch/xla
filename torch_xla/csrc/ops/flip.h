#ifndef XLA_TORCH_XLA_CSRC_OPS_FLIP_H_
#define XLA_TORCH_XLA_CSRC_OPS_FLIP_H_

#include "absl/types/span.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Flip : public XlaNode {
 public:
  Flip(const torch::lazy::Value& input, std::vector<int64_t> dims);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& dims() const { return dims_; }

 private:
  // The dimensions which are flipped.
  std::vector<int64_t> dims_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_FLIP_H_