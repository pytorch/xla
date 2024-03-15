#ifndef XLA_TORCH_XLA_CSRC_OPS_RESIZE_H_
#define XLA_TORCH_XLA_CSRC_OPS_RESIZE_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Resize : public XlaNode {
 public:
  Resize(const torch::lazy::Value& input, std::vector<int64_t> size);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& size() const { return size_; }

 private:
  std::vector<int64_t> size_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_RESIZE_H_
