#ifndef XLA_TORCH_XLA_CSRC_OPS_ROLL_H_
#define XLA_TORCH_XLA_CSRC_OPS_ROLL_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Roll : public XlaNode {
 public:
  Roll(const torch::lazy::Value& input, std::vector<int64_t> shifts,
       std::vector<int64_t> dims);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& shifts() const { return shifts_; }

  const std::vector<int64_t>& dims() const { return dims_; }

 private:
  std::vector<int64_t> shifts_;
  std::vector<int64_t> dims_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_ROLL_H_