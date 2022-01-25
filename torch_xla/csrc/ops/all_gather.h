#pragma once

#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class AllGather : public Node {
 public:
  AllGather(const Value& input, const Value& token, xla::int64_t dim,
            xla::int64_t shard_count,
            std::vector<std::vector<xla::int64_t>> groups);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  xla::int64_t dim() const { return dim_; }

  xla::int64_t shard_count() const { return shard_count_; }

  const std::vector<std::vector<xla::int64_t>>& groups() const {
    return groups_;
  }

 private:
  xla::int64_t dim_;
  xla::int64_t shard_count_;
  std::vector<std::vector<xla::int64_t>> groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
