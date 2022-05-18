#include "torch_xla/csrc/ops/ops_xla_shape_fn.h"

#include "torch_xla/csrc/helpers.h"

namespace torch_xla {

xla::Shape AbsOutputShape(const XlaValue& input) { return input.xla_shape(); }

xla::Shape AcosOutputShape(const XlaValue& input) { return input.xla_shape(); }

xla::Shape AcoshOutputShape(const XlaValue& input) { return input.xla_shape(); }

xla::Shape AsinOutputShape(const XlaValue& input) { return input.xla_shape(); }

xla::Shape AsinhOutputShape(const XlaValue& input) { return input.xla_shape(); }

xla::Shape AtanOutputShape(const XlaValue& input) { return input.xla_shape(); }

xla::Shape AtanhOutputShape(const XlaValue& input) { return input.xla_shape(); }

xla::Shape CosOutputShape(const XlaValue& input) { return input.xla_shape(); }

xla::Shape CoshOutputShape(const XlaValue& input) { return input.xla_shape(); }

xla::Shape LogOutputShape(const XlaValue& input) { return input.xla_shape(); }

xla::Shape Log2OutputShape(const XlaValue& input) { return input.xla_shape(); }

xla::Shape Log10OutputShape(const XlaValue& input) { return input.xla_shape(); }

xla::Shape MaximumOutputShape(const XlaValue& input, const XlaValue& other) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    auto promoted = XlaHelpers::Promote(operands[0], operands[1]);
    return xla::Max(promoted.first, promoted.second);
  };
  return InferOutputShape({input.xla_shape(), other.xla_shape()},
                          lower_for_shape_fn);
}

xla::Shape SgnOutputShape(const XlaValue& input) { return input.xla_shape(); }

xla::Shape SignOutputShape(const XlaValue& input) { return input.xla_shape(); }

}  // namespace torch_xla
