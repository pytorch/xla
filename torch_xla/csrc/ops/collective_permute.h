#pragma once

#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class CollectivePermute : public Node {
 public:
  CollectivePermute(
      const Value& input, const Value& token,
      std::vector<std::pair<xla::int64, xla::int64>> source_target_pairs);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<std::pair<xla::int64, xla::int64>>& source_target_pairs()
      const {
    return source_target_pairs_;
  }

 private:
  std::vector<std::pair<xla::int64, xla::int64>> source_target_pairs_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
