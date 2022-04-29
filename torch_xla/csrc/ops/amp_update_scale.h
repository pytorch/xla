#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class AmpUpdateScale : public XlaNode {
 public:
  AmpUpdateScale(const XlaValue& current_scale, const XlaValue& growth_tracker,
                 const XlaValue& found_inf, double scale_growth_factor,
                 double scale_backoff_factor, int growth_interval);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  double scale_growth_factor_;
  double scale_backoff_factor_;
  int growth_interval_;
};

}  // namespace torch_xla
