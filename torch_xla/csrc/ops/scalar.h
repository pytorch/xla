#pragma once

#include "ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

// Differently from Constant, this is a scalar value broadcasted to a shape.
// Even though a Constant could have been used, for simple scalars broadcasted
// to big shapes, the Constant leads to big literals expanded within the XLA
// graph.
class Scalar : public Node {
 public:
  Scalar(double value, xla::Shape shape);
  Scalar(double value, xla::PrimitiveType type);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  double value_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
