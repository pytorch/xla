#pragma once

#include "absl/types/span.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Roll : public Node {
 public:
  Roll(const Value& input, std::vector<int64_t> shifts,
       std::vector<int64_t> dims);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& shifts() const { return shifts_; }

  const std::vector<int64_t>& dims() const { return dims_; }

 private:
  std::vector<int64_t> shifts_;
  std::vector<int64_t> dims_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
