#ifndef XLA_TORCH_XLA_CSRC_OPS_AMP_UPDATE_SCALE_H_
#define XLA_TORCH_XLA_CSRC_OPS_AMP_UPDATE_SCALE_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class AmpUpdateScale : public XlaNode {
 public:
  AmpUpdateScale(const torch::lazy::Value& current_scale,
                 const torch::lazy::Value& growth_tracker,
                 const torch::lazy::Value& found_inf,
                 double scale_growth_factor, double scale_backoff_factor,
                 int growth_interval);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  double scale_growth_factor_;
  double scale_backoff_factor_;
  int growth_interval_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_AMP_UPDATE_SCALE_H_