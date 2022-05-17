#pragma once

// This header can depend on ops/ and ir.h, as well as system/c++, tensorflow,
// PT,... but not on other PT/XLA headers.

#include <memory>

#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ops/constant.h"
#include "torch_xla/csrc/ops/generic.h"
#include "torch_xla/csrc/ops/scalar.h"

namespace torch_xla {

inline torch::lazy::NodePtr ScalarOp(const at::Scalar& value,
                                     xla::Shape shape) {
  return torch::lazy::MakeNode<Scalar>(value, std::move(shape));
}
inline torch::lazy::NodePtr ScalarOp(const at::Scalar& value,
                                     xla::PrimitiveType type) {
  return torch::lazy::MakeNode<Scalar>(value, type);
}

inline torch::lazy::NodePtr ConstantOp(xla::Literal value) {
  return torch::lazy::MakeNode<Constant>(std::move(value));
}

inline torch::lazy::NodePtr GenericOp(
    torch::lazy::OpKind op, absl::Span<const XlaValue> operands,
    xla::Shape shape, Generic::LowerFn lower_fn, size_t num_outputs = 1,
    // cast to uint32_t to avoid ambiguous constructor of uint128
    torch::lazy::hash_t hash_seed = (uint32_t)0x5a2d296e9) {
  return torch::lazy::MakeNode<Generic>(std::move(op), operands,
                                        std::move(shape), std::move(lower_fn),
                                        num_outputs, hash_seed);
}

inline torch::lazy::NodePtr GenericOp(
    torch::lazy::OpKind op, absl::Span<const XlaValue> operands,
    const std::function<xla::Shape()>& shape_fn, Generic::LowerFn lower_fn,
    size_t num_outputs = 1,
    // cast to uint32_t to avoid ambiguous constructor of uint128
    torch::lazy::hash_t hash_seed = (uint32_t)0x5a2d296e9) {
  return torch::lazy::MakeNode<Generic>(std::move(op), operands, shape_fn,
                                        std::move(lower_fn), num_outputs,
                                        hash_seed);
}

inline torch::lazy::NodePtr GenericOp(torch::lazy::OpKind op, xla::Shape shape,
                                      Generic::LowerFn lower_fn,
                                      size_t num_outputs,
                                      torch::lazy::hash_t hash_seed) {
  return torch::lazy::MakeNode<Generic>(std::move(op), std::move(shape),
                                        std::move(lower_fn), num_outputs,
                                        hash_seed);
}

torch::lazy::NodePtr Sin(const XlaValue& input);

torch::lazy::NodePtr Sinh(const XlaValue& input);

torch::lazy::NodePtr Atan2(const XlaValue& input, const XlaValue& other);

torch::lazy::NodePtr Tan(const XlaValue& input);

torch::lazy::NodePtr Tanh(const XlaValue& input);

torch::lazy::NodePtr Neg(const XlaValue& input);

torch::lazy::NodePtr ReluOp(const XlaValue& input);

torch::lazy::NodePtr Min(const XlaValue& input, const XlaValue& other);

torch::lazy::NodePtr Exp(const XlaValue& input);

torch::lazy::NodePtr Expm1(const XlaValue& input);

torch::lazy::NodePtr Erf(const XlaValue& input);

torch::lazy::NodePtr Erfc(const XlaValue& input);

torch::lazy::NodePtr Erfinv(const XlaValue& input);

torch::lazy::NodePtr Log1p(const XlaValue& input);

torch::lazy::NodePtr Sqrt(const XlaValue& input);

torch::lazy::NodePtr Rsqrt(const XlaValue& input);

torch::lazy::NodePtr ReciprocalOp(const XlaValue& input);

torch::lazy::NodePtr Prelu(const XlaValue& input, const XlaValue& weight);

torch::lazy::NodePtr Pow(const XlaValue& input, const XlaValue& exponent);

torch::lazy::NodePtr Fmod(const XlaValue& dividend, const XlaValue& divisor);

torch::lazy::NodePtr Not(const XlaValue& input);

torch::lazy::NodePtr HardSigmoid(const XlaValue& input);

torch::lazy::NodePtr HardSigmoidBackward(const XlaValue& grad_output,
                                         const XlaValue& input);

torch::lazy::NodePtr HardSwish(const XlaValue& input);

torch::lazy::NodePtr HardSwishBackward(const XlaValue& grad_output,
                                       const XlaValue& input);

torch::lazy::NodePtr LogSigmoid(const XlaValue& input);

torch::lazy::NodePtr LogSigmoidBackward(const XlaValue& grad_output,
                                        const XlaValue& input,
                                        const XlaValue& buffer);

torch::lazy::NodePtr Sigmoid(const XlaValue& input);

torch::lazy::NodePtr SiLU(const XlaValue& input);

torch::lazy::NodePtr SiLUBackward(const XlaValue& grad_output,
                                  const XlaValue& input);

torch::lazy::NodePtr SigmoidBackward(const XlaValue& grad_output,
                                     const XlaValue& output);

torch::lazy::NodePtr LogSoftmaxBackwardOp(const XlaValue& grad_output,
                                          const XlaValue& output, int64_t dim);

torch::lazy::NodePtr SoftmaxBackwardOp(const XlaValue& grad_output,
                                       const XlaValue& output, int64_t dim);

torch::lazy::NodePtr Clamp(const XlaValue& input, const XlaValue& min,
                           const XlaValue& max);

torch::lazy::NodePtr Ceil(const XlaValue& input);

torch::lazy::NodePtr Celu(const XlaValue& input, const at::Scalar& alpha);

torch::lazy::NodePtr Floor(const XlaValue& input);

torch::lazy::NodePtr Round(const XlaValue& input);

torch::lazy::NodePtr Trunc(const XlaValue& input);

torch::lazy::NodePtr FracOp(const XlaValue& input);

torch::lazy::NodePtr Ger(const XlaValue& input, const XlaValue& other);

torch::lazy::NodePtr AddMatMulOp(const XlaValue& input, const XlaValue& weight,
                                 const XlaValue& bias);

torch::lazy::NodePtr Dot(const XlaValue& input, const XlaValue& weight);

torch::lazy::NodePtr MatMul(const XlaValue& lhs, const XlaValue& rhs);

torch::lazy::NodePtr AdaptiveMaxPool2dBackward(const XlaValue& grad_output,
                                               const XlaValue& input);

torch::lazy::NodePtr AdaptiveAvgPool2dBackward(const XlaValue& grad_output,
                                               const XlaValue& input);

torch::lazy::NodePtr AdaptiveAvgPool3dBackward(const XlaValue& grad_output,
                                               const XlaValue& input);

torch::lazy::NodePtr ComparisonOp(c10::Symbol kind, const XlaValue& input,
                                  const XlaValue& other);

torch::lazy::NodePtr Where(const XlaValue& condition, const XlaValue& input,
                           const XlaValue& other);

torch::lazy::NodePtr ARange(const at::Scalar& start, const at::Scalar& end,
                            const at::Scalar& step, at::ScalarType scalar_type);

torch::lazy::NodePtr BroadcastTensors(absl::Span<const XlaValue> tensors);

torch::lazy::NodePtr Norm(const XlaValue& input,
                          const c10::optional<at::Scalar>& p,
                          c10::optional<at::ScalarType> dtype,
                          absl::Span<const int64_t> dims, bool keepdim);

torch::lazy::NodePtr Identity(int64_t lines, int64_t cols,
                              xla::PrimitiveType element_type);

torch::lazy::NodePtr Elu(const XlaValue& input, const at::Scalar& alpha,
                         const at::Scalar& scale,
                         const at::Scalar& input_scale);

torch::lazy::NodePtr EluBackward(const XlaValue& grad_output,
                                 const XlaValue& output,
                                 const at::Scalar& alpha,
                                 const at::Scalar& scale,
                                 const at::Scalar& input_scale);

torch::lazy::NodePtr Gelu(const XlaValue& input);

torch::lazy::NodePtr GeluBackward(const XlaValue& grad, const XlaValue& input);

torch::lazy::NodePtr Lshift(const XlaValue& input, const at::Scalar& other);

torch::lazy::NodePtr Lshift(const XlaValue& input, const XlaValue& other);

torch::lazy::NodePtr Rshift(const XlaValue& input, const at::Scalar& other);

torch::lazy::NodePtr Rshift(const XlaValue& input, const XlaValue& other);

torch::lazy::NodePtr Remainder(const XlaValue& input, const XlaValue& divisor);

torch::lazy::NodePtr MaxUnary(const XlaValue& input);

torch::lazy::NodePtr MinUnary(const XlaValue& input);

torch::lazy::NodePtr Take(const XlaValue& input, const XlaValue& index);

torch::lazy::NodePtr TanhGelu(const XlaValue& input);

torch::lazy::NodePtr TanhGeluBackward(const XlaValue& grad,
                                      const XlaValue& input);

torch::lazy::NodePtr LogDet(const XlaValue& input);

torch::lazy::NodePtr Inverse(const XlaValue& input);

torch::lazy::NodePtr IsNan(const XlaValue& input);

torch::lazy::NodePtr BaddBmm(const XlaValue& lhs, const XlaValue& rhs,
                             const XlaValue& bias,
                             const XlaValue& product_multiplier,
                             const XlaValue& bias_multiplier);

torch::lazy::NodePtr Lerp(const XlaValue& start, const XlaValue& end,
                          const XlaValue& weight);

torch::lazy::NodePtr LogicalNot(const XlaValue& input);

torch::lazy::NodePtr LogicalXor(const XlaValue& input, const XlaValue& other);

torch::lazy::NodePtr LogicalAnd(const XlaValue& input, const XlaValue& other);

torch::lazy::NodePtr LogicalOr(const XlaValue& input, const XlaValue& other);

torch::lazy::NodePtr XLogY(const XlaValue& input, const XlaValue& other);

torch::lazy::NodePtr NanToNum(const XlaValue& input, const XlaValue& nan,
                              const XlaValue& posinf, const XlaValue& neginf);

torch::lazy::NodePtr SLogDet(const XlaValue& input);

torch::lazy::NodePtr Softplus(const XlaValue& input, const XlaValue& beta,
                              const XlaValue& threshold);

torch::lazy::NodePtr Selu(const XlaValue& input);

}  // namespace torch_xla
