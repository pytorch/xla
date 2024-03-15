#ifndef XLA_TORCH_XLA_CSRC_OPS_ALL_REDUCE_H_
#define XLA_TORCH_XLA_CSRC_OPS_ALL_REDUCE_H_

#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class AllReduce : public XlaNode {
 public:
  AllReduce(AllReduceType reduce_type,
            c10::ArrayRef<torch::lazy::Value> operands,
            const torch::lazy::Value& token, double scale,
            std::vector<std::vector<int64_t>> groups, bool pin_layout);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

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

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_ALL_REDUCE_H_