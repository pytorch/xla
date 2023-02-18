#pragma once

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

// IR node for 2D & 3D convolutions with or without bias.
class Unique2 : public XlaNode {
 public:
  Unique2(const torch::lazy::Value& input, bool sorted, bool return_inverse,
          bool return_counts);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

 private:
  bool sorted_;
  bool return_inverse_;
  bool return_counts_;
};

}  // namespace torch_xla
