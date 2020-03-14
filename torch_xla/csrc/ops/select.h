#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Select : public Node {
 public:
  Select(const Value& input, xla::int64 dim, xla::int64 start, xla::int64 end,
         xla::int64 stride);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 dim() const { return dim_; }

  xla::int64 start() const { return start_; }

  xla::int64 end() const { return end_; }

  xla::int64 stride() const { return stride_; }

  static xla::Shape MakeSelectShape(const xla::Shape& shape, xla::int64 dim,
                                    xla::int64 start, xla::int64 end,
                                    xla::int64 stride);

  static xla::int64 GetStride(xla::int64 start, xla::int64 end,
                              xla::int64 stride);

 private:
  xla::int64 dim_;
  xla::int64 start_;
  xla::int64 end_;
  xla::int64 stride_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
