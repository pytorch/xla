#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

// IR node for log(softmax) operation.
class LogSoftmax : public Node {
 public:
  LogSoftmax(const Value& input, xla::int64 dim,
             c10::optional<at::ScalarType> dtype);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 dim() const { return dim_; }

  const c10::optional<at::ScalarType>& dtype() const { return dtype_; }

 private:
  // The dimension along which the result is computed.
  xla::int64 dim_;
  c10::optional<at::ScalarType> dtype_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
