#pragma once

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class ExpandDynamic : public XlaNode {
 public:
  ExpandDynamic(const torch::lazy::Value& input,
                const std::vector<torch::lazy::Value>& dimensions,
                const std::vector<int64_t> upper_bounds,
                const std::vector<bool> dynamic_dims);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<int64_t>& size() const { return upper_bounds_; };

  const bool IsDynamic(int index) const { return dynamic_dims_[index]; };

 private:
  std::vector<int64_t> upper_bounds_;
  std::vector<bool> dynamic_dims_;
  std::vector<xla::Shape> shapes_;
};

}  // namespace torch_xla
