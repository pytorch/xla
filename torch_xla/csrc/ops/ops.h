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

inline NodePtr ScalarOp(const at::Scalar& value, xla::Shape shape) {
  return MakeNode<Scalar>(value, std::move(shape));
}
inline NodePtr ScalarOp(const at::Scalar& value, xla::PrimitiveType type) {
  return MakeNode<Scalar>(value, type);
}

inline NodePtr ConstantOp(xla::Literal value) {
  return MakeNode<Constant>(std::move(value));
}

inline NodePtr GenericOp(
    OpKind op, absl::Span<const Value> operands, xla::Shape shape,
    Generic::LowerFn lower_fn, size_t num_outputs = 1,
    // cast to uint32_t to avoid ambiguous constructor of uint128
    torch::lazy::hash_t hash_seed = (uint32_t)0x5a2d296e9) {
  return MakeNode<Generic>(std::move(op), operands, std::move(shape),
                           std::move(lower_fn), num_outputs, hash_seed);
}

inline NodePtr GenericOp(
    OpKind op, absl::Span<const Value> operands,
    const std::function<xla::Shape()>& shape_fn, Generic::LowerFn lower_fn,
    size_t num_outputs = 1,
    // cast to uint32_t to avoid ambiguous constructor of uint128
    torch::lazy::hash_t hash_seed = (uint32_t)0x5a2d296e9) {
  return MakeNode<Generic>(std::move(op), operands, shape_fn,
                           std::move(lower_fn), num_outputs, hash_seed);
}

inline NodePtr GenericOp(OpKind op, xla::Shape shape, Generic::LowerFn lower_fn,
                         size_t num_outputs, torch::lazy::hash_t hash_seed) {
  return MakeNode<Generic>(std::move(op), std::move(shape), std::move(lower_fn),
                           num_outputs, hash_seed);
}

NodePtr Acos(const Value& input);

NodePtr Acosh(const Value& input);

NodePtr Cos(const Value& input);

NodePtr Cosh(const Value& input);

NodePtr Asin(const Value& input);

NodePtr Asinh(const Value& input);

NodePtr Sin(const Value& input);

NodePtr Sinh(const Value& input);

NodePtr Atan(const Value& input);

NodePtr Atanh(const Value& input);

NodePtr Atan2(const Value& input, const Value& other);

NodePtr Tan(const Value& input);

NodePtr Tanh(const Value& input);

NodePtr Neg(const Value& input);

NodePtr SgnOp(const Value& input);

NodePtr SignOp(const Value& input);

NodePtr Abs(const Value& input);

NodePtr ReluOp(const Value& input);

NodePtr Min(const Value& input, const Value& other);

NodePtr Max(const Value& input, const Value& other);

NodePtr Exp(const Value& input);

NodePtr Expm1(const Value& input);

NodePtr Erf(const Value& input);

NodePtr Erfc(const Value& input);

NodePtr Erfinv(const Value& input);

NodePtr Log(const Value& input);

NodePtr LogBase(const Value& input, OpKind op, double base);

NodePtr Log1p(const Value& input);

NodePtr Sqrt(const Value& input);

NodePtr Rsqrt(const Value& input);

NodePtr ReciprocalOp(const Value& input);

NodePtr Prelu(const Value& input, const Value& weight);

NodePtr Pow(const Value& input, const Value& exponent);

NodePtr Fmod(const Value& dividend, const Value& divisor);

NodePtr Not(const Value& input);

NodePtr HardSigmoid(const Value& input);

NodePtr HardSigmoidBackward(const Value& grad_output, const Value& input);

std::tuple<NodePtr, NodePtr> LogSigmoid(const Value& input);

NodePtr LogSigmoidBackward(const Value& grad_output, const Value& input,
                           const Value& buffer);

NodePtr Sigmoid(const Value& input);

NodePtr SiLU(const Value& input);

NodePtr SiLUBackward(const Value& grad_output, const Value& input);

NodePtr SigmoidBackward(const Value& grad_output, const Value& output);

NodePtr LogSoftmaxBackwardOp(const Value& grad_output, const Value& output,
                             int64_t dim);

NodePtr SoftmaxBackwardOp(const Value& grad_output, const Value& output,
                          int64_t dim);

NodePtr Clamp(const Value& input, const Value& min, const Value& max);

NodePtr Ceil(const Value& input);

NodePtr Floor(const Value& input);

NodePtr Round(const Value& input);

NodePtr Trunc(const Value& input);

NodePtr FracOp(const Value& input);

NodePtr Ger(const Value& input, const Value& other);

NodePtr AddMatMulOp(const Value& input, const Value& weight, const Value& bias);

NodePtr Dot(const Value& input, const Value& weight);

NodePtr MatMul(const Value& lhs, const Value& rhs);

NodePtr AdaptiveMaxPool2dBackward(const Value& grad_output, const Value& input);

NodePtr AdaptiveAvgPool2dBackward(const Value& grad_output, const Value& input);

NodePtr AdaptiveAvgPool3dBackward(const Value& grad_output, const Value& input);

NodePtr ComparisonOp(c10::Symbol kind, const Value& input, const Value& other);

NodePtr Where(const Value& condition, const Value& input, const Value& other);

NodePtr ARange(const at::Scalar& start, const at::Scalar& end,
               const at::Scalar& step, at::ScalarType scalar_type);

NodePtr BroadcastTensors(absl::Span<const Value> tensors);

NodePtr Norm(const Value& input, const c10::optional<at::Scalar>& p,
             c10::optional<at::ScalarType> dtype,
             absl::Span<const int64_t> dims, bool keepdim);

NodePtr Identity(int64_t lines, int64_t cols,
                 xla::PrimitiveType element_type);

NodePtr Elu(const Value& input, const at::Scalar& alpha,
            const at::Scalar& scale, const at::Scalar& input_scale);

NodePtr EluBackward(const Value& grad_output, const Value& output,
                    const at::Scalar& alpha, const at::Scalar& scale,
                    const at::Scalar& input_scale);

NodePtr Gelu(const Value& input);

NodePtr GeluBackward(const Value& grad, const Value& input);

NodePtr Lshift(const Value& input, const at::Scalar& other);

NodePtr Lshift(const Value& input, const Value& other);

NodePtr Rshift(const Value& input, const at::Scalar& other);

NodePtr Rshift(const Value& input, const Value& other);

NodePtr Remainder(const Value& input, const Value& divisor);

NodePtr MaxUnary(const Value& input);

NodePtr MinUnary(const Value& input);

NodePtr Take(const Value& input, const Value& index);

NodePtr LogDet(const Value& input);

NodePtr Inverse(const Value& input);

NodePtr IsNan(const Value& input);

NodePtr BaddBmm(const Value& lhs, const Value& rhs, const Value& bias,
                const Value& product_multiplier, const Value& bias_multiplier);

NodePtr Lerp(const Value& start, const Value& end, const Value& weight);

NodePtr LogicalNot(const Value& input);

NodePtr LogicalXor(const Value& input, const Value& other);

NodePtr LogicalAnd(const Value& input, const Value& other);

NodePtr LogicalOr(const Value& input, const Value& other);

NodePtr XLogY(const Value& input, const Value& other);

NodePtr NanToNum(const Value& input, const Value& nan, const Value& posinf,
                 const Value& neginf);

NodePtr SLogDet(const Value& input);

NodePtr Softplus(const Value& input, const Value& beta, const Value& threshold);

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
