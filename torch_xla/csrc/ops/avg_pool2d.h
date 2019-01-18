#pragma once

#include "ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class AvgPool2d : public Node {
 public:
  AvgPool2d(const NodeOperand& input, int kernel_size, int stride, int padding,
            bool count_include_pad);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int kernel_size() const { return kernel_size_; }

  int stride() const { return stride_; }

  int padding() const { return padding_; }

  bool count_include_pad() const { return count_include_pad_; }

 private:
  // The parameters of the pooling. Only support the same kernel size, stride
  // and padding in both dimensions for now. Ceil mode not supported yet.
  int kernel_size_;
  int stride_;
  int padding_;
  // Whether the counts used to compute the average should include the added
  // padding.
  bool count_include_pad_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
