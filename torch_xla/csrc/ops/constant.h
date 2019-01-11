#pragma once

#include "ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Constant : public Node {
 public:
  Constant(xla::Literal value);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  xla::Literal value_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
