#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class MaxPoolNd : public Node {
 public:
  MaxPoolNd(const Value& input, xla::int64 spatial_dim_count,
            std::vector<xla::int64> kernel_size, std::vector<xla::int64> stride,
            std::vector<xla::int64> padding, bool ceil_mode);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 spatial_dim_count() const { return spatial_dim_count_; }

  const std::vector<xla::int64>& kernel_size() const { return kernel_size_; }

  const std::vector<xla::int64>& stride() const { return stride_; }

  const std::vector<xla::int64>& padding() const { return padding_; }

  bool ceil_mode() const { return ceil_mode_; }

 private:
  xla::int64 spatial_dim_count_;
  // The parameters of the pooling. Only support the same kernel size, stride
  // and padding in both dimensions for now.
  std::vector<xla::int64> kernel_size_;
  std::vector<xla::int64> stride_;
  std::vector<xla::int64> padding_;
  bool ceil_mode_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
