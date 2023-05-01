#ifndef XLA_TORCH_XLA_CSRC_OPS_NOT_SUPPORTED_H_
#define XLA_TORCH_XLA_CSRC_OPS_NOT_SUPPORTED_H_

#include <string>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class NotSupported : public XlaNode {
 public:
  NotSupported(std::string description, xla::Shape shape);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::string& description() const { return description_; }

 private:
  std::string description_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_NOT_SUPPORTED_H_
