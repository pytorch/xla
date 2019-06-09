#pragma once

#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class AdaptiveAvgPool2d : public Node {
 public:
  AdaptiveAvgPool2d(const Value& input, std::vector<xla::int64> output_size);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& output_size() const { return output_size_; }

 private:
  std::vector<xla::int64> output_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
