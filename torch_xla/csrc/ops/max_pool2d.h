#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

// IR node for 2D max pooling.
class MaxPool2d : public Node {
 public:
  MaxPool2d(const Value& input,
            tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
            tensorflow::gtl::ArraySlice<const xla::int64> stride,
            tensorflow::gtl::ArraySlice<const xla::int64> padding);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& kernel_size() const { return kernel_size_; }

  const std::vector<xla::int64>& stride() const { return stride_; }

  const std::vector<xla::int64>& padding() const { return padding_; }

 private:
  // The parameters of the pooling. Only support the same kernel size, stride
  // and padding in both dimensions for now.
  std::vector<xla::int64> kernel_size_;
  std::vector<xla::int64> stride_;
  std::vector<xla::int64> padding_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
