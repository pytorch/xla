#pragma once

#include "absl/types/span.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Repeat : public Node {
 public:
  Repeat(const Value& input, std::vector<int64_t> repeats);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& repeats() const { return repeats_; }

 private:
  // The number of repeats along each dimension.
  std::vector<int64_t> repeats_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
