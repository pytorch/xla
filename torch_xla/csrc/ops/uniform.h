#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Uniform : public Node {
 public:
  Uniform(const Value& from, const Value& to, const Value& seed,
          const xla::Shape& rng_shape);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
