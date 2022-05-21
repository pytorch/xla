#include "torch_xla/csrc/ops/ops_xla_shape_fn.h"

#include "torch_xla/csrc/helpers.h"

namespace torch_xla {

xla::Shape AbsOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape AcosOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape AcoshOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape AsinOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape AsinhOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape AtanOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape AtanhOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape CosOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape CoshOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape MaximumOutputShape(const torch::lazy::Value& input,
                              const torch::lazy::Value& other) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    auto promoted = XlaHelpers::Promote(operands[0], operands[1]);
    return xla::Max(promoted.first, promoted.second);
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(other)},
                          lower_for_shape_fn);
}

xla::Shape MinimumOutputShape(const torch::lazy::Value& input,
                              const torch::lazy::Value& other) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    auto promoted = XlaHelpers::Promote(operands[0], operands[1]);
    return xla::Max(promoted.first, promoted.second);
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(other)},
                          lower_for_shape_fn);
}

xla::Shape SgnOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape SignOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape SinOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape SinhOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape TanOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

}  // namespace torch_xla
