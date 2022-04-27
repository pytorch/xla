#pragma once

#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class AllToAll : public XlaNode {
 public:
  AllToAll(const XlaValue& input, const XlaValue& token,
           int64_t split_dimension, int64_t concat_dimension,
           int64_t split_count, std::vector<std::vector<int64_t>> groups,
           bool pin_layout);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t split_dimension() const { return split_dimension_; }

  int64_t concat_dimension() const { return concat_dimension_; }

  int64_t split_count() const { return split_count_; }

  const std::vector<std::vector<int64_t>>& groups() const { return groups_; }

  bool pin_layout() const { return pin_layout_; }

 private:
  int64_t split_dimension_;
  int64_t concat_dimension_;
  int64_t split_count_;
  std::vector<std::vector<int64_t>> groups_;
  bool pin_layout_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
