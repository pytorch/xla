#pragma once

#include "ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class CrossReplicaSum : public Node {
 public:
  CrossReplicaSum(const NodeOperand& operand,
                  std::vector<std::vector<xla::int64>> groups);

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  std::vector<std::vector<xla::int64>> groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
