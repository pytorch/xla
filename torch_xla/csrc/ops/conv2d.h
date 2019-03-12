#pragma once

#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

// IR node for 2D convolutions with or without bias.
class Conv2d : public Node {
 public:
  Conv2d(const Value& input, const Value& weight, const Value& bias,
         std::vector<xla::int64> stride, std::vector<xla::int64> padding);

  Conv2d(const Value& input, const Value& weight,
         std::vector<xla::int64> stride, std::vector<xla::int64> padding);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& stride() const { return stride_; }

  const std::vector<xla::int64>& padding() const { return padding_; }

 private:
  // The parameters of the convolution.
  std::vector<xla::int64> stride_;
  std::vector<xla::int64> padding_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
