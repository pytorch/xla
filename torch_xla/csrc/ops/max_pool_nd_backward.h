#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class MaxPoolNdBackward : public Node {
 public:
  MaxPoolNdBackward(const Value& grad_output, const Value& input,
                    xla::int64 spatial_dim_count,
                    std::vector<xla::int64> kernel_size,
                    std::vector<xla::int64> stride,
                    std::vector<xla::int64> padding);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 spatial_dim_count() const { return spatial_dim_count_; }

  const std::vector<xla::int64>& kernel_size() const { return kernel_size_; }

  const std::vector<xla::int64>& stride() const { return stride_; }

  const std::vector<xla::int64>& padding() const { return padding_; }

 private:
  xla::int64 spatial_dim_count_;
  // The parameters of the pooling. Ceil mode not supported yet.
  std::vector<xla::int64> kernel_size_;
  std::vector<xla::int64> stride_;
  std::vector<xla::int64> padding_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
