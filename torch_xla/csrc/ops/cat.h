#ifndef XLA_TORCH_XLA_CSRC_OPS_CAT_H_
#define XLA_TORCH_XLA_CSRC_OPS_CAT_H_

#include <c10/core/ScalarType.h>

#include "absl/types/span.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Cat : public XlaNode {
 public:
  Cat(c10::ArrayRef<torch::lazy::Value> values, int64_t dim,
      at::ScalarType dtype);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t dim() const { return dim_; };

  at::ScalarType dtype() const { return dtype_; };

 private:
  int64_t dim_;
  at::ScalarType dtype_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_CAT_H_