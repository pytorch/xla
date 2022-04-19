#pragma once

// This header can depend on ops/ and ir.h, as well as system/c++, tensorflow,
// PT,... but not on other PT/XLA headers.

#include <memory>

#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ops/constant.h"
#include "torch_xla/csrc/ops/generic.h"
#include "torch_xla/csrc/ops/scalar.h"

namespace torch_xla {
namespace ir {
namespace ops {

inline torch::lazy::NodePtr ScalarOp(const at::Scalar& value,
                                     xla::Shape shape) {
  return ir::MakeNode<Scalar>(value, std::move(shape));
}
inline torch::lazy::NodePtr ScalarOp(const at::Scalar& value,
                                     xla::PrimitiveType type) {
  return ir::MakeNode<Scalar>(value, type);
}

inline torch::lazy::NodePtr ConstantOp(xla::Literal value) {
  return ir::MakeNode<Constant>(std::move(value));
}

inline torch::lazy::NodePtr GenericOp(
    torch::lazy::OpKind op, absl::Span<const Value> operands, xla::Shape shape,
    Generic::LowerFn lower_fn, size_t num_outputs = 1,
    // cast to uint32_t to avoid ambiguous constructor of uint128
    torch::lazy::hash_t hash_seed = (uint32_t)0x5a2d296e9) {
  return torch_xla::ir::MakeNode<Generic>(std::move(op), operands,
                                          std::move(shape), std::move(lower_fn),
                                          num_outputs, hash_seed);
}

inline torch::lazy::NodePtr GenericOp(
    torch::lazy::OpKind op, absl::Span<const Value> operands,
    const std::function<xla::Shape()>& shape_fn, Generic::LowerFn lower_fn,
    size_t num_outputs = 1,
    // cast to uint32_t to avoid ambiguous constructor of uint128
    torch::lazy::hash_t hash_seed = (uint32_t)0x5a2d296e9) {
  return torch_xla::ir::MakeNode<Generic>(std::move(op), operands, shape_fn,
                                          std::move(lower_fn), num_outputs,
                                          hash_seed);
}

inline torch::lazy::NodePtr GenericOp(torch::lazy::OpKind op, xla::Shape shape,
                                      Generic::LowerFn lower_fn,
                                      size_t num_outputs,
                                      torch::lazy::hash_t hash_seed) {
  return torch_xla::ir::MakeNode<Generic>(std::move(op), std::move(shape),
                                          std::move(lower_fn), num_outputs,
                                          hash_seed);
}

torch::lazy::NodePtr Acos(const Value& input);

torch::lazy::NodePtr Acosh(const Value& input);

torch::lazy::NodePtr Cos(const Value& input);

torch::lazy::NodePtr Cosh(const Value& input);

torch::lazy::NodePtr Asin(const Value& input);

torch::lazy::NodePtr Asinh(const Value& input);

torch::lazy::NodePtr Sin(const Value& input);

torch::lazy::NodePtr Sinh(const Value& input);

torch::lazy::NodePtr Atan(const Value& input);

torch::lazy::NodePtr Atanh(const Value& input);

torch::lazy::NodePtr Atan2(const Value& input, const Value& other);

torch::lazy::NodePtr Tan(const Value& input);

torch::lazy::NodePtr Tanh(const Value& input);

torch::lazy::NodePtr Neg(const Value& input);

torch::lazy::NodePtr SgnOp(const Value& input);

torch::lazy::NodePtr SignOp(const Value& input);

torch::lazy::NodePtr Abs(const Value& input);

torch::lazy::NodePtr ReluOp(const Value& input);

torch::lazy::NodePtr Min(const Value& input, const Value& other);

torch::lazy::NodePtr Max(const Value& input, const Value& other);

torch::lazy::NodePtr Exp(const Value& input);

torch::lazy::NodePtr Expm1(const Value& input);

torch::lazy::NodePtr Erf(const Value& input);

torch::lazy::NodePtr Erfc(const Value& input);

torch::lazy::NodePtr Erfinv(const Value& input);

torch::lazy::NodePtr Log(const Value& input);

torch::lazy::NodePtr LogBase(const Value& input, torch::lazy::OpKind op,
                             double base);

torch::lazy::NodePtr Log1p(const Value& input);

torch::lazy::NodePtr Sqrt(const Value& input);

torch::lazy::NodePtr Rsqrt(const Value& input);

torch::lazy::NodePtr ReciprocalOp(const Value& input);

torch::lazy::NodePtr Prelu(const Value& input, const Value& weight);

torch::lazy::NodePtr Pow(const Value& input, const Value& exponent);

torch::lazy::NodePtr Fmod(const Value& dividend, const Value& divisor);

torch::lazy::NodePtr Not(const Value& input);

torch::lazy::NodePtr HardSigmoid(const Value& input);

torch::lazy::NodePtr HardSigmoidBackward(const Value& grad_output,
                                         const Value& input);

torch::lazy::NodePtr HardSwish(const Value& input);

torch::lazy::NodePtr HardSwishBackward(const Value& grad_output,
                                       const Value& input);

std::tuple<torch::lazy::NodePtr, torch::lazy::NodePtr> LogSigmoid(
    const Value& input);

torch::lazy::NodePtr LogSigmoidBackward(const Value& grad_output,
                                        const Value& input,
                                        const Value& buffer);

torch::lazy::NodePtr Sigmoid(const Value& input);

torch::lazy::NodePtr SiLU(const Value& input);

torch::lazy::NodePtr SiLUBackward(const Value& grad_output, const Value& input);

torch::lazy::NodePtr SigmoidBackward(const Value& grad_output,
                                     const Value& output);

torch::lazy::NodePtr LogSoftmaxBackwardOp(const Value& grad_output,
                                          const Value& output, int64_t dim);

torch::lazy::NodePtr SoftmaxBackwardOp(const Value& grad_output,
                                       const Value& output, int64_t dim);

torch::lazy::NodePtr Clamp(const Value& input, const Value& min,
                           const Value& max);

torch::lazy::NodePtr Ceil(const Value& input);

torch::lazy::NodePtr Floor(const Value& input);

torch::lazy::NodePtr Round(const Value& input);

torch::lazy::NodePtr Trunc(const Value& input);

torch::lazy::NodePtr FracOp(const Value& input);

torch::lazy::NodePtr Ger(const Value& input, const Value& other);

torch::lazy::NodePtr AddMatMulOp(const Value& input, const Value& weight,
                                 const Value& bias);

torch::lazy::NodePtr Dot(const Value& input, const Value& weight);

torch::lazy::NodePtr MatMul(const Value& lhs, const Value& rhs);

torch::lazy::NodePtr AdaptiveMaxPool2dBackward(const Value& grad_output,
                                               const Value& input);

torch::lazy::NodePtr AdaptiveAvgPool2dBackward(const Value& grad_output,
                                               const Value& input);

torch::lazy::NodePtr AdaptiveAvgPool3dBackward(const Value& grad_output,
                                               const Value& input);

torch::lazy::NodePtr ComparisonOp(c10::Symbol kind, const Value& input,
                                  const Value& other);

torch::lazy::NodePtr Where(const Value& condition, const Value& input,
                           const Value& other);

torch::lazy::NodePtr ARange(const at::Scalar& start, const at::Scalar& end,
                            const at::Scalar& step, at::ScalarType scalar_type);

torch::lazy::NodePtr BroadcastTensors(absl::Span<const Value> tensors);

torch::lazy::NodePtr Norm(const Value& input,
                          const c10::optional<at::Scalar>& p,
                          c10::optional<at::ScalarType> dtype,
                          absl::Span<const int64_t> dims, bool keepdim);

torch::lazy::NodePtr Identity(int64_t lines, int64_t cols,
                              xla::PrimitiveType element_type);

torch::lazy::NodePtr Elu(const Value& input, const at::Scalar& alpha,
                         const at::Scalar& scale,
                         const at::Scalar& input_scale);

torch::lazy::NodePtr EluBackward(const Value& grad_output, const Value& output,
                                 const at::Scalar& alpha,
                                 const at::Scalar& scale,
                                 const at::Scalar& input_scale);

torch::lazy::NodePtr Gelu(const Value& input);

torch::lazy::NodePtr GeluBackward(const Value& grad, const Value& input);

torch::lazy::NodePtr Lshift(const Value& input, const at::Scalar& other);

torch::lazy::NodePtr Lshift(const Value& input, const Value& other);

torch::lazy::NodePtr Rshift(const Value& input, const at::Scalar& other);

torch::lazy::NodePtr Rshift(const Value& input, const Value& other);

torch::lazy::NodePtr Remainder(const Value& input, const Value& divisor);

torch::lazy::NodePtr MaxUnary(const Value& input);

torch::lazy::NodePtr MinUnary(const Value& input);

torch::lazy::NodePtr Take(const Value& input, const Value& index);

torch::lazy::NodePtr TanhGelu(const Value& input);

torch::lazy::NodePtr TanhGeluBackward(const Value& grad, const Value& input);

torch::lazy::NodePtr LogDet(const Value& input);

torch::lazy::NodePtr Inverse(const Value& input);

torch::lazy::NodePtr IsNan(const Value& input);

torch::lazy::NodePtr BaddBmm(const Value& lhs, const Value& rhs,
                             const Value& bias, const Value& product_multiplier,
                             const Value& bias_multiplier);

torch::lazy::NodePtr Lerp(const Value& start, const Value& end,
                          const Value& weight);

torch::lazy::NodePtr LogicalNot(const Value& input);

torch::lazy::NodePtr LogicalXor(const Value& input, const Value& other);

torch::lazy::NodePtr LogicalAnd(const Value& input, const Value& other);

torch::lazy::NodePtr LogicalOr(const Value& input, const Value& other);

torch::lazy::NodePtr XLogY(const Value& input, const Value& other);

torch::lazy::NodePtr NanToNum(const Value& input, const Value& nan,
                              const Value& posinf, const Value& neginf);

torch::lazy::NodePtr SLogDet(const Value& input);

torch::lazy::NodePtr Softplus(const Value& input, const Value& beta,
                              const Value& threshold);

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
