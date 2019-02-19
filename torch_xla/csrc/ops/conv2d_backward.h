#pragma once

#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Conv2dBackward : public Node {
 public:
  Conv2dBackward(const Value& grad_output, const Value& input,
                 const Value& weight,
                 tensorflow::gtl::ArraySlice<const xla::int64> stride,
                 tensorflow::gtl::ArraySlice<const xla::int64> padding);

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
