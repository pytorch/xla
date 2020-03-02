#pragma once

#include <c10/util/Optional.h>

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Normal : public Node {
 public:
  Normal(const Value& mean, const Value& std, xla::uint64 seed);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  xla::uint64 seed_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
