#ifndef XLA_TORCH_XLA_CSRC_OPS_COUNT_NONZERO_H_
#define XLA_TORCH_XLA_CSRC_OPS_COUNT_NONZERO_H_

//#include <c10/core/ScalarType.h>
//#include <c10/util/Optional.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class CountNonzero : public XlaNode {
 public:
  CountNonzero(const torch::lazy::Value& input, c10::optional<int64_t> dim);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  c10::optional<int64_t> dim() const { return dim_; }

 private:
  int64_t dim_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_COUNT_NONZERO_H_
