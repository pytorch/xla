#pragma once

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

// IR node for 2D & 3D convolutions with or without bias.
class ConvolutionOverrideable : public Node {
 public:
  ConvolutionOverrideable(const Value& input, const Value& weight,
                          const Value& bias, std::vector<int64_t> stride,
                          std::vector<int64_t> padding,
                          std::vector<int64_t> dilation, bool transposed,
                          std::vector<int64_t> output_padding,
                          int64_t groups);

  ConvolutionOverrideable(const Value& input, const Value& weight,
                          std::vector<int64_t> stride,
                          std::vector<int64_t> padding,
                          std::vector<int64_t> dilation, bool transposed,
                          std::vector<int64_t> output_padding,
                          int64_t groups);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& stride() const { return stride_; }

  const std::vector<int64_t>& padding() const { return padding_; }

  const std::vector<int64_t>& dilation() const { return dilation_; }

  bool transposed() const { return transposed_; }

  const std::vector<int64_t>& output_padding() const {
    return output_padding_;
  }

  int64_t groups() const { return groups_; }

 private:
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> dilation_;
  std::vector<int64_t> output_padding_;
  bool transposed_;
  int64_t groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
