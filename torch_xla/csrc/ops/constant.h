#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Constant : public Node {
 public:
  Constant(xla::Literal value);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const xla::Literal& value() const { return value_; }

 private:
  xla::Literal value_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
