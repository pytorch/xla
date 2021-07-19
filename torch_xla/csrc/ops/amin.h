#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Amin : public Node {
 public:
  Amin(const Value& input, std::vector<xla::int64> dimensions, bool keepdim);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::vector<xla::int64> dim() const { return dimensions_; };

  bool keepdim() const { return keepdim_; }

 private:
  std::vector<xla::int64> dimensions_;
  bool keepdim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
