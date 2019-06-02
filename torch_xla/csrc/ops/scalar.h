#pragma once

#include <c10/core/Scalar.h>

#include <iostream>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

inline std::ostream& operator<<(std::ostream& ostrm, at::Scalar s) {
  return ostrm << (s.isFloatingPoint() ? s.toDouble() : s.toLong());
}

inline size_t ScalarHash(at::Scalar s) {
  return s.isFloatingPoint() ? std::hash<double>()(s.toDouble())
                             : std::hash<long>()(s.toLong());
}

// Differently from Constant, this is a scalar value broadcasted to a shape.
// Even though a Constant could have been used, for simple scalars broadcasted
// to big shapes, the Constant leads to big literals expanded within the XLA
// graph.
class Scalar : public Node {
 public:
  Scalar(at::Scalar value, xla::Shape shape);
  Scalar(at::Scalar value, xla::PrimitiveType type);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  at::Scalar value() const { return value_; }

 private:
  at::Scalar value_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
