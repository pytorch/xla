#ifndef XLA_TORCH_XLA_CSRC_OPS_CUMPROD_H_
#define XLA_TORCH_XLA_CSRC_OPS_CUMPROD_H_

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class CumProd : public XlaNode {
 public:
  CumProd(const torch::lazy::Value& input, int64_t dim,
          c10::optional<at::ScalarType> dtype);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t dim() const { return dim_; }

  const c10::optional<at::ScalarType>& dtype() const { return dtype_; }

 private:
  int64_t dim_;
  c10::optional<at::ScalarType> dtype_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_CUMPROD_H_