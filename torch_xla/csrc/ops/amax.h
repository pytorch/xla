#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Amax : public Node {
 public:
  Amax(const Value& input, std::vector<int64_t> dimensions, bool keepdim);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::vector<int64_t> dim() const { return dimensions_; };

  bool keepdim() const { return keepdim_; }

 private:
  std::vector<int64_t> dimensions_;
  bool keepdim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
