#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Uniform : public Node {
 public:
  Uniform(const Value& from, const Value& to, const xla::Shape& rng_shape,
          xla::uint64 seed);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  xla::uint64 seed() const { return seed_; }

 private:
  xla::uint64 seed_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
