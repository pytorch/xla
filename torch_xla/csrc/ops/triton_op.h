#ifndef XLA_TORCH_XLA_CSRC_OPS_TRITON_OP_H_
#define XLA_TORCH_XLA_CSRC_OPS_TRITON_OP_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class TritonOp : public XlaNode {
 public:
  TritonOp(std::vector<const torch::lazy::Value> inputs, std::vector<float> fparams, std::vector<int> iparams, std::string name);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string name() const { return name_; };

 private:
  std::string name_;
  int num_inputs_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_TRITON_OP_H_