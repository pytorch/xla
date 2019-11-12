#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class CrossReplicaSum : public Node {
 public:
  CrossReplicaSum(tensorflow::gtl::ArraySlice<const Value> operands,
                  const Value& token, double scale,
                  std::vector<std::vector<xla::int64>> groups);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  double scale() const { return scale_; }

  const std::vector<std::vector<xla::int64>>& groups() const { return groups_; }

 private:
  double scale_;
  std::vector<std::vector<xla::int64>> groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
