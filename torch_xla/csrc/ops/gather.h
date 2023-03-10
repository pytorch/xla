#ifndef XLA_TORCH_XLA_CSRC_OPS_GATHER_H_
#define XLA_TORCH_XLA_CSRC_OPS_GATHER_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Gather : public XlaNode {
 public:
  Gather(const torch::lazy::Value& input, int64_t dim,
         const torch::lazy::Value& index);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t dim() const { return dim_; };

 private:
  int64_t dim_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_GATHER_H_