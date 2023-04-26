#ifndef XLA_TORCH_XLA_CSRC_OPS_MAX_UNPOOL_ND_H_
#define XLA_TORCH_XLA_CSRC_OPS_MAX_UNPOOL_ND_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class MaxUnpoolNd : public XlaNode {
 public:
  MaxUnpoolNd(const torch::lazy::Value& input,
              const torch::lazy::Value& indices,
              std::vector<int64_t> output_size);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& output_size() const { return output_size_; }

 private:
  std::vector<int64_t> output_size_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_MAX_UNPOOL_ND_H_