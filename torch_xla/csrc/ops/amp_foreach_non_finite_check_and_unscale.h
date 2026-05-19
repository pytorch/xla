#ifndef XLA_TORCH_XLA_CSRC_OPS_AMP_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_H_
#define XLA_TORCH_XLA_CSRC_OPS_AMP_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class AmpForachNonFiniteCheckAndUnscale : public XlaNode {
 public:
  AmpForachNonFiniteCheckAndUnscale(const torch::lazy::OpList& inputs,
                                    const torch::lazy::Value& found_inf,
                                    const torch::lazy::Value& inv_scale);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_AMP_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_H_