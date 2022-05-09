#include "torch_xla/csrc/ops/ops_xla_shape_fn.h"

#include "torch_xla/csrc/helpers.h"

namespace torch_xla {

xla::Shape AbsOutputShape(const XlaValue& input) { return input.xla_shape(); }

xla::Shape MaximumOutputShape(const XlaValue& input, const XlaValue& other) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    auto promoted = XlaHelpers::Promote(operands[0], operands[1]);
    return xla::Max(promoted.first, promoted.second);
  };
  return InferOutputShape({input.xla_shape(), other.xla_shape()},
                          lower_for_shape_fn);
}

}  // namespace torch_xla
