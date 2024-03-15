#ifndef XLA_TORCH_XLA_CSRC_OPS_STACK_H_
#define XLA_TORCH_XLA_CSRC_OPS_STACK_H_

#include "absl/types/span.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Stack : public XlaNode {
 public:
  Stack(c10::ArrayRef<torch::lazy::Value> values, int64_t dim);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t dim() const { return dim_; };

 private:
  int64_t dim_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_STACK_H_