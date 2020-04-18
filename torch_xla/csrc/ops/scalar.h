#pragma once

#include <c10/core/Scalar.h>

#include <iostream>

#include "tensorflow/compiler/xla/xla_client/types.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

// Differently from Constant, this is a scalar value broadcasted to a shape.
// Even though a Constant could have been used, for simple scalars broadcasted
// to big shapes, the Constant leads to big literals expanded within the XLA
// graph.
class Scalar : public Node {
 public:
  Scalar(at::Scalar value, xla::Shape shape);
  Scalar(at::Scalar value, xla::PrimitiveType type);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const at::Scalar& value() const { return value_; }

 private:
  at::Scalar value_;
};

xla::hash_t ScalarHash(at::Scalar s);

std::ostream& operator<<(std::ostream& ostrm, at::Scalar s);

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
