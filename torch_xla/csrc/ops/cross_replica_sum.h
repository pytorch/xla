#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class CrossReplicaSum : public Node {
 public:
  CrossReplicaSum(const Value& operand,
                  std::vector<std::vector<xla::int64>> groups);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  std::vector<std::vector<xla::int64>> groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
