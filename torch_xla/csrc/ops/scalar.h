#ifndef XLA_TORCH_XLA_CSRC_OPS_SCALAR_H_
#define XLA_TORCH_XLA_CSRC_OPS_SCALAR_H_

#include <ATen/core/Formatting.h>
#include <c10/core/Scalar.h>

#include <iostream>

#include "third_party/xla_client/types.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

// Differently from Constant, this is a scalar value broadcasted to a shape.
// Even though a Constant could have been used, for simple scalars broadcasted
// to big shapes, the Constant leads to big literals expanded within the XLA
// graph.
class Scalar : public XlaNode {
 public:
  Scalar(const at::Scalar& value, xla::Shape shape);
  Scalar(const at::Scalar& value, xla::PrimitiveType type);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const at::Scalar& value() const { return value_; }

 private:
  at::Scalar value_;
};

torch::lazy::hash_t ScalarHash(const at::Scalar& s);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_SCALAR_H_