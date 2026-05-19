#ifndef XLA_TORCH_XLA_CSRC_OPS_MARK_TENSOR_H_
#define XLA_TORCH_XLA_CSRC_OPS_MARK_TENSOR_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class MarkTensor : public XlaNode {
 public:
  MarkTensor(const torch::lazy::Value& input, const std::string& info);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  std::string info_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_MARK_TENSOR_H_
