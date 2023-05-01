#ifndef XLA_TORCH_XLA_CSRC_OPS_CONSTANT_PAD_ND_H_
#define XLA_TORCH_XLA_CSRC_OPS_CONSTANT_PAD_ND_H_

#include <c10/core/Scalar.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class ConstantPadNd : public XlaNode {
 public:
  ConstantPadNd(const torch::lazy::Value& input, std::vector<int64_t> pad,
                const at::Scalar& value);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const at::Scalar& value() const { return value_; }

  const std::vector<int64_t>& pad() const { return pad_; }

 private:
  std::vector<int64_t> pad_;
  at::Scalar value_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_CONSTANT_PAD_ND_H_