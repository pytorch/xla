#ifndef XLA_TORCH_XLA_CSRC_OPS_SPLIT_H_
#define XLA_TORCH_XLA_CSRC_OPS_SPLIT_H_

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

// Split the tensor into chunks along a given dimension.
class Split : public XlaNode {
 public:
  Split(const torch::lazy::Value& input, std::vector<int64_t> split_sizes,
        int64_t dim);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& split_sizes() const { return split_sizes_; }

  int64_t dim() const { return dim_; }

 private:
  std::vector<int64_t> split_sizes_;
  int64_t dim_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_SPLIT_H_#pragma once