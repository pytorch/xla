#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class AmpUpdateScale : public Node {
 public:
  AmpUpdateScale(const Value& current_scale, const Value& growth_tracker,
                 const Value& found_inf, double scale_growth_factor,
                 double scale_backoff_factor, int growth_interval);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  double scale_growth_factor_;
  double scale_backoff_factor_;
  int growth_interval_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
