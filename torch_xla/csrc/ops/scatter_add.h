#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class ScatterAdd : public Node {
 public:
  ScatterAdd(const Value& input, const Value& index, const Value& src,
             xla::int64 dim);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  xla::int64 dim() const { return dim_; };

 private:
  xla::int64 dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
