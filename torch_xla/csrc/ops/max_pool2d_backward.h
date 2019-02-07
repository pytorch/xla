#pragma once

#include "ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class MaxPool2dBackward : public Node {
 public:
  MaxPool2dBackward(const Value& grad_output, const Value& input,
                    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
                    tensorflow::gtl::ArraySlice<const xla::int64> stride,
                    tensorflow::gtl::ArraySlice<const xla::int64> padding);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& kernel_size() const { return kernel_size_; }

  const std::vector<xla::int64>& stride() const { return stride_; }

  const std::vector<xla::int64>& padding() const { return padding_; }

 private:
  // The parameters of the pooling. Ceil mode not supported yet.
  std::vector<xla::int64> kernel_size_;
  std::vector<xla::int64> stride_;
  std::vector<xla::int64> padding_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
