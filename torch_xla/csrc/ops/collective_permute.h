#ifndef XLA_TORCH_XLA_CSRC_OPS_COLLECTIVE_PERMUTE_H_
#define XLA_TORCH_XLA_CSRC_OPS_COLLECTIVE_PERMUTE_H_

#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class CollectivePermute : public XlaNode {
 public:
  CollectivePermute(
      const torch::lazy::Value& input, const torch::lazy::Value& token,
      std::vector<std::pair<int64_t, int64_t>> source_target_pairs);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs() const {
    return source_target_pairs_;
  }

 private:
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_COLLECTIVE_PERMUTE_H_