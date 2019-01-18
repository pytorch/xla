#pragma once

#include "ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

// IR node for log(softmax) operation.
class LogSoftmax : public Node {
 public:
  LogSoftmax(const NodeOperand& input, xla::int64 dim);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

 private:
  // The dimension along which the result is computed.
  xla::int64 dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
