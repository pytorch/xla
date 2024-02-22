#ifndef XLA_TORCH_XLA_CSRC_OPS_EMBEDDING_BAG_H_
#define XLA_TORCH_XLA_CSRC_OPS_EMBEDDING_BAG_H_

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class EmbeddingBag : public XlaNode {
 public:
  EmbeddingBag(const torch::lazy::Value& weight,
               const torch::lazy::Value& indices,
               const torch::lazy::Value& offsets, int64_t mode,
               const torch::lazy::Value& per_sample_weights,
               bool include_last_offset);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  int64_t mode_;
  bool include_last_offset_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_EMBEDDING_BAG_H_