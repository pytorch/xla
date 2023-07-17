#ifndef XLA_TORCH_XLA_CSRC_OPS_RRELU_WITH_NOISE_H_
#define XLA_TORCH_XLA_CSRC_OPS_RRELU_WITH_NOISE_H_

#include <c10/core/Scalar.h>

#include "torch_xla/csrc/ir.h"
#include "xla/types.h"

namespace torch_xla {

class RreluWithNoise : public XlaNode {
 public:
  RreluWithNoise(const torch::lazy::Value& input,
                 const torch::lazy::Value& seed, const at::Scalar& lower,
                 const at::Scalar& upper, bool training);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const at::Scalar& lower() const { return lower_; }

  const at::Scalar& upper() const { return upper_; }

  bool training() const { return training_; }

 private:
  at::Scalar lower_;
  at::Scalar upper_;
  bool training_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_RRELU_WITH_NOISE_H_