#pragma once

#include <vector>
#include <iostream>
#include <c10/core/Scalar.h>

#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/tensor.h"
#include "tensorflow/compiler/xla/xla_client/types.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Fill : public Node {
 public:
  Fill(const Value& input,
       const std::vector<xla::int64> size,
       const at::Scalar& value,
       const Device& device,
       const std::shared_ptr<xla::ComputationClient::Data> data,
       const xla::PrimitiveType type,
       const xla::Shape& shape);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<xla::int64>& size() const { return size_; };

 private:
  const std::vector<xla::int64> size_;
  const at::Scalar value_;
  const Device& device_;
  const std::shared_ptr<xla::ComputationClient::Data> data_;
  const xla::PrimitiveType type_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
