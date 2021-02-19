#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class AmpForachNonFiniteCheckAndUnscale : public Node {
 public:
  AmpForachNonFiniteCheckAndUnscale(const OpList& inputs,
                                    const Value& found_inf,
                                    const Value& inv_scale);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
