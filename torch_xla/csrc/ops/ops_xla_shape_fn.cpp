#include "torch_xla/csrc/ops/ops_xla_shape_fn.h"

#include "tensorflow/compiler/xla/client/lib/logdet.h"
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

xla::Shape ErfOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape ErfcOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape ErfinvOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape ExpOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape FloorOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape HardsigmoidOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape HardsigmoidBackwardOutputShape(const torch::lazy::Value& grad_output,
                                          const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape HardswishOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape HardswishBackwardOutputShape(const torch::lazy::Value& grad_output,
                                        const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape InverseOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape LogdetOutputShape(const torch::lazy::Value& input) {
  const xla::Shape& input_shape = GetXlaShape(input);
  XLA_CHECK_GE(input_shape.rank(), 2) << input_shape;
  // The input tensor is ...,N,N
  xla::Shape logdet_shape(input_shape);
  logdet_shape.DeleteDimension(input_shape.rank() - 1);
  logdet_shape.DeleteDimension(input_shape.rank() - 2);
  return logdet_shape;
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

xla::Shape ReciprocalOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape RsqrtOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
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

/* Blocked on https://github.com/pytorch/xla/issues/3596 */
// xla::Shape SlogdetOutputShape(const torch::lazy::Value& input) {
//   auto lower_for_shape_fn =
//       [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
//     xla::SignAndLogDet result = xla::SLogDet(operands[0]);
//     return xla::Tuple(operands[0].builder(), {result.sign, result.logdet});
//   };
//   return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
// }

xla::Shape TanOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape TanhOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

}  // namespace torch_xla
