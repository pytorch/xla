#pragma once

#include "ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

// IR node for 2D max pooling.
class MaxPool2d : public Node {
 public:
  MaxPool2d(const NodeOperand& input, int kernel_size, int stride, int padding);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int kernel_size() const { return kernel_size_; }

  int stride() const { return stride_; }

  int padding() const { return padding_; }

 private:
  // The parameters of the pooling. Only support the same kernel size, stride
  // and padding in both dimensions for now.
  int kernel_size_;
  int stride_;
  int padding_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
