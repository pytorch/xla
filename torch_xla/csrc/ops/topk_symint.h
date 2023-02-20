#pragma once

#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {

class TopKSymInt : public XlaNode {
 public:
  TopKSymInt(const torch::lazy::Value& input, const SymIntElements& k,
             int64_t dim, bool largest, bool sorted, bool stable);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  int64_t k_upper_bound_;
  int64_t dim_;
  bool largest_;
  bool sorted_;
  bool stable_;
};

}  // namespace torch_xla
