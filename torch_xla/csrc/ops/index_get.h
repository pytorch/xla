#ifndef XLA_TORCH_XLA_CSRC_OPS_INDEX_GET_H_
#define XLA_TORCH_XLA_CSRC_OPS_INDEX_GET_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class IndexGet : public XlaNode {
 public:
  IndexGet(const torch::lazy::Value& base, const torch::lazy::Value& indices,
           int64_t start_dim);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t start_dim() const { return start_dim_; }

 private:
  // The dimension number at which indexing starts.
  int64_t start_dim_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_INDEX_GET_H_