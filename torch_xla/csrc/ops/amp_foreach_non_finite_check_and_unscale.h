#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class AmpForachNonFiniteCheckAndUnscale : public XlaNode {
 public:
  AmpForachNonFiniteCheckAndUnscale(const OpList& inputs,
                                    const XlaValue& found_inf,
                                    const XlaValue& inv_scale);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace torch_xla
