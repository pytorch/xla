#pragma once

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class ConvolutionBackwardOverrideable : public Node {
 public:
  ConvolutionBackwardOverrideable(
      const Value& grad_output, const Value& input, const Value& weight,
      std::vector<xla::int64> stride, std::vector<xla::int64> padding,
      std::vector<xla::int64> dilation, bool transposed,
      std::vector<xla::int64> output_padding, xla::int64 groups);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& stride() const { return stride_; }

  const std::vector<xla::int64>& padding() const { return padding_; }

  const std::vector<xla::int64>& dilation() const { return dilation_; }

  bool transposed() const { return transposed_; }

  const std::vector<xla::int64>& output_padding() const {
    return output_padding_;
  }

  xla::int64 groups() const { return groups_; }

 private:
  std::vector<xla::int64> stride_;
  std::vector<xla::int64> padding_;
  std::vector<xla::int64> dilation_;
  std::vector<xla::int64> output_padding_;
  bool transposed_;
  xla::int64 groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
