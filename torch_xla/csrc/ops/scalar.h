#pragma once

#include <ATen/core/Formatting.h>
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
  Scalar(const at::Scalar& value, xla::Shape shape);
  Scalar(const at::Scalar& value, xla::PrimitiveType type);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const at::Scalar& value() const { return value_; }

 private:
  at::Scalar value_;
};

torch::lazy::hash_t ScalarHash(const at::Scalar& s);

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
