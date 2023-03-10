#ifndef XLA_TORCH_XLA_CSRC_OPS_CDIST_H_
#define XLA_TORCH_XLA_CSRC_OPS_CDIST_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class CdistForward : public XlaNode {
 public:
  CdistForward(const torch::lazy::Value& x1, const torch::lazy::Value& x2,
               const torch::lazy::Value& p, bool use_hamming,
               bool use_chebyshev);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  bool use_hamming_;    // handle p == 0
  bool use_chebyshev_;  // handle p == +inf
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_CDIST_H_