#ifndef XLA_TORCH_XLA_CSRC_OPS_MAX_POOL_ND_H_
#define XLA_TORCH_XLA_CSRC_OPS_MAX_POOL_ND_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class MaxPoolNd : public XlaNode {
 public:
  MaxPoolNd(const torch::lazy::Value& input, int64_t spatial_dim_count,
            std::vector<int64_t> kernel_size, std::vector<int64_t> stride,
            std::vector<int64_t> padding, bool ceil_mode);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int64_t spatial_dim_count() const { return spatial_dim_count_; }

  const std::vector<int64_t>& kernel_size() const { return kernel_size_; }

  const std::vector<int64_t>& stride() const { return stride_; }

  const std::vector<int64_t>& padding() const { return padding_; }

  bool ceil_mode() const { return ceil_mode_; }

 private:
  int64_t spatial_dim_count_;
  // The parameters of the pooling. Only support the same kernel size, stride
  // and padding in both dimensions for now.
  std::vector<int64_t> kernel_size_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  bool ceil_mode_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_MAX_POOL_ND_H_