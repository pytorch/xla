#pragma once

#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class ReduceScatter : public Node {
 public:
  ReduceScatter(AllReduceType reduce_type, const Value& input,
                const Value& token, double scale, int64_t scatter_dim,
                int64_t shard_count, std::vector<std::vector<int64_t>> groups);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  AllReduceType reduce_type() const { return reduce_type_; }

  double scale() const { return scale_; }

  const std::vector<std::vector<int64_t>>& groups() const { return groups_; }

 private:
  AllReduceType reduce_type_;
  double scale_;
  int64_t scatter_dim_;
  int64_t shard_count_;
  std::vector<std::vector<int64_t>> groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
