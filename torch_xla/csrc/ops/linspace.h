#ifndef XLA_TORCH_XLA_CSRC_OPS_LINSPACE_H_
#define XLA_TORCH_XLA_CSRC_OPS_LINSPACE_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Linspace : public XlaNode {
 public:
  Linspace(const torch::lazy::Value& start, const torch::lazy::Value& end,
           const int64_t steps);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t steps() const { return steps_; };

 private:
  int64_t steps_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_LINSPACE_H_