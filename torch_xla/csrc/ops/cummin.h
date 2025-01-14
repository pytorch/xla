#ifndef XLA_TORCH_XLA_CSRC_OPS_CUMMIN_H_
#define XLA_TORCH_XLA_CSRC_OPS_CUMMIN_H_

#include <c10/core/ScalarType.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class CumMax : public XlaNode {
 public:
  CumMin(const torch::lazy::Value& input, int64_t dim);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t dim() const { return dim_; }

 private:
  int64_t dim_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_CUMMIN_H_
