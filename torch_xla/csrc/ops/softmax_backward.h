#pragma once

#include "ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class LogSoftmaxBackward : public Node {
 public:
  LogSoftmaxBackward(const Value& grad_output, const Value& output,
                     xla::int64 dim);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 dim() const { return dim_; }

 private:
  // The dimension along which the result is computed.
  xla::int64 dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
