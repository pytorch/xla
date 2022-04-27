#pragma once

#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class AllReduce : public XlaNode {
 public:
  AllReduce(AllReduceType reduce_type, absl::Span<const XlaValue> operands,
            const XlaValue& token, double scale,
            std::vector<std::vector<int64_t>> groups, bool pin_layout);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  AllReduceType reduce_type() const { return reduce_type_; }

  double scale() const { return scale_; }

  const std::vector<std::vector<int64_t>>& groups() const { return groups_; }

  bool pin_layout() const { return pin_layout_; }

 private:
  AllReduceType reduce_type_;
  double scale_;
  std::vector<std::vector<int64_t>> groups_;
  bool pin_layout_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
