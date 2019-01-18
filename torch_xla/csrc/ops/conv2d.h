#pragma once

#include "ir.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace torch_xla {
namespace ir {
namespace ops {

// IR node for 2D convolutions with or without bias.
class Conv2d : public Node {
 public:
  Conv2d(const NodeOperand& input, const NodeOperand& weight,
         const NodeOperand& bias, int stride, int padding,
         bool use_full_conv_precision);

  Conv2d(const NodeOperand& input, const NodeOperand& weight, int stride,
         int padding, bool use_full_conv_precision);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int stride() const { return stride_; }

  int padding() const { return padding_; }

  xla::PrecisionConfig::Precision precision() const { return precision_; }

 private:
  // The parameters of the convolution. Only support the same stride and padding
  // in both dimension for now.
  int stride_;
  int padding_;
  // The numeric precision to use on TPU.
  xla::PrecisionConfig::Precision precision_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
