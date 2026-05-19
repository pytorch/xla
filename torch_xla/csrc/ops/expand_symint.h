#ifndef XLA_TORCH_XLA_CSRC_OPS_EXPAND_SYMINT_H_
#define XLA_TORCH_XLA_CSRC_OPS_EXPAND_SYMINT_H_

#include <vector>

#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {

class ExpandSymInt : public XlaNode {
 public:
  ExpandSymInt(const torch::lazy::Value& input,
               const SymIntElements& size_elements);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<int64_t>& size() const { return upper_bounds_; };

  const bool IsDynamic(int index) const { return dynamic_dims_[index]; };

 private:
  std::vector<int64_t> upper_bounds_;
  std::vector<bool> dynamic_dims_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_EXPAND_SYMINT_H_