#ifndef XLA_TORCH_XLA_CSRC_OPS_ALL_GATHER_H_
#define XLA_TORCH_XLA_CSRC_OPS_ALL_GATHER_H_

#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class AllGather : public XlaNode {
 public:
  AllGather(const torch::lazy::Value& input, const torch::lazy::Value& token,
            int64_t dim, int64_t shard_count,
            std::vector<std::vector<int64_t>> groups, bool pin_layout);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t dim() const { return dim_; }

  int64_t shard_count() const { return shard_count_; }

  const std::vector<std::vector<int64_t>>& groups() const { return groups_; }

  bool pin_layout() const { return pin_layout_; }

 private:
  int64_t dim_;
  int64_t shard_count_;
  std::vector<std::vector<int64_t>> groups_;
  bool pin_layout_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_ALL_GATHER_H_