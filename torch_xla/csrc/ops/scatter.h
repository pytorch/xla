#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Scatter : public Node {
 public:
  Scatter(const Value& input, const Value& index, const Value& src,
          int64_t dim);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t dim() const { return dim_; };

 private:
  int64_t dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
