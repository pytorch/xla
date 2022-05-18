#pragma once

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class ViewOp : public XlaNode {
 public:
  ViewOp(const torch::lazy::Value& input, std::vector<int64_t> output_size);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& output_size() const { return output_size_; }

 private:
  std::vector<int64_t> output_size_;
};

}  // namespace torch_xla
