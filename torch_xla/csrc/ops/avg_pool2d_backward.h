#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class AvgPool2dBackward : public Node {
 public:
  AvgPool2dBackward(const Value& grad_output, const Value& input,
                    std::vector<xla::int64> kernel_size,
                    std::vector<xla::int64> stride,
                    std::vector<xla::int64> padding, bool count_include_pad);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& kernel_size() const { return kernel_size_; }

  const std::vector<xla::int64>& stride() const { return stride_; }

  const std::vector<xla::int64>& padding() const { return padding_; }

  bool count_include_pad() const { return count_include_pad_; }

 private:
  // The parameters of the pooling. Ceil mode not supported yet.
  std::vector<xla::int64> kernel_size_;
  std::vector<xla::int64> stride_;
  std::vector<xla::int64> padding_;
  // Whether the counts used to compute the average should include the added
  // padding.
  bool count_include_pad_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
