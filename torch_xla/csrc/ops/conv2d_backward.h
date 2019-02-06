#pragma once

#include "ir.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Conv2dBackward : public Node {
 public:
  Conv2dBackward(const NodeOperand& grad_output, const NodeOperand& input,
                 const NodeOperand& weight,
                 tensorflow::gtl::ArraySlice<const xla::int64> stride,
                 tensorflow::gtl::ArraySlice<const xla::int64> padding,
                 bool use_full_conv_precision);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& stride() const { return stride_; }

  const std::vector<xla::int64>& padding() const { return padding_; }

  xla::PrecisionConfig::Precision precision() const { return precision_; }

 private:
  // The parameters of the convolution.
  std::vector<xla::int64> stride_;
  std::vector<xla::int64> padding_;
  // The numeric precision to use on TPU.
  xla::PrecisionConfig::Precision precision_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
