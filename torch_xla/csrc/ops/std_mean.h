#pragma once

#include <vector>

#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class StdMean : public XlaNode {
 public:
  StdMean(const torch::lazy::Value& input, std::vector<int64_t> dimensions,
          int64_t correction, bool keep_reduced_dimensions);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<int64_t>& dimensions() const { return dimensions_; }

  int64_t correction() const { return correction_; }

  bool keep_reduced_dimensions() const { return keep_reduced_dimensions_; }

 private:
  std::vector<int64_t> dimensions_;
  int64_t correction_;
  bool keep_reduced_dimensions_;
};

}  // namespace torch_xla
