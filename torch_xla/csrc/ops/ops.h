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
    torch::lazy::OpKind op, c10::ArrayRef<torch::lazy::Value> operands,
    xla::Shape shape, Generic::LowerFn lower_fn, size_t num_outputs = 1,
    // cast to uint32_t to avoid ambiguous constructor of uint128
    torch::lazy::hash_t hash_seed = (uint32_t)0x5a2d296e9) {
  return torch::lazy::MakeNode<Generic>(std::move(op), operands,
                                        std::move(shape), std::move(lower_fn),
                                        num_outputs, hash_seed);
}

inline torch::lazy::NodePtr GenericOp(
    torch::lazy::OpKind op, c10::ArrayRef<torch::lazy::Value> operands,
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

torch::lazy::NodePtr Cos(const torch::lazy::Value& input);

torch::lazy::NodePtr Cosh(const torch::lazy::Value& input);

torch::lazy::NodePtr Sin(const torch::lazy::Value& input);

torch::lazy::NodePtr Sinh(const torch::lazy::Value& input);

torch::lazy::NodePtr Atan2(const torch::lazy::Value& input,
                           const torch::lazy::Value& other);

torch::lazy::NodePtr Tan(const torch::lazy::Value& input);

torch::lazy::NodePtr Tanh(const torch::lazy::Value& input);

torch::lazy::NodePtr Neg(const torch::lazy::Value& input);

torch::lazy::NodePtr SgnOp(const torch::lazy::Value& input);

torch::lazy::NodePtr SignOp(const torch::lazy::Value& input);

torch::lazy::NodePtr ReluOp(const torch::lazy::Value& input);

torch::lazy::NodePtr Min(const torch::lazy::Value& input,
                         const torch::lazy::Value& other);

torch::lazy::NodePtr Exp(const torch::lazy::Value& input);

torch::lazy::NodePtr Expm1(const torch::lazy::Value& input);

torch::lazy::NodePtr Erf(const torch::lazy::Value& input);

torch::lazy::NodePtr Erfc(const torch::lazy::Value& input);

torch::lazy::NodePtr Erfinv(const torch::lazy::Value& input);

torch::lazy::NodePtr Log(const torch::lazy::Value& input);

torch::lazy::NodePtr LogBase(const torch::lazy::Value& input,
                             torch::lazy::OpKind op, double base);

torch::lazy::NodePtr Log1p(const torch::lazy::Value& input);

torch::lazy::NodePtr Sqrt(const torch::lazy::Value& input);

torch::lazy::NodePtr Rsqrt(const torch::lazy::Value& input);

torch::lazy::NodePtr Prelu(const torch::lazy::Value& input,
                           const torch::lazy::Value& weight);

torch::lazy::NodePtr Pow(const torch::lazy::Value& input,
                         const torch::lazy::Value& exponent);

torch::lazy::NodePtr Fmod(const torch::lazy::Value& dividend,
                          const torch::lazy::Value& divisor);

torch::lazy::NodePtr Not(const torch::lazy::Value& input);

torch::lazy::NodePtr HardSigmoid(const torch::lazy::Value& input);

torch::lazy::NodePtr HardSigmoidBackward(const torch::lazy::Value& grad_output,
                                         const torch::lazy::Value& input);

torch::lazy::NodePtr HardSwish(const torch::lazy::Value& input);

torch::lazy::NodePtr HardSwishBackward(const torch::lazy::Value& grad_output,
                                       const torch::lazy::Value& input);

torch::lazy::NodePtr LogSigmoid(const torch::lazy::Value& input);

torch::lazy::NodePtr LogSigmoidBackward(const torch::lazy::Value& grad_output,
                                        const torch::lazy::Value& input,
                                        const torch::lazy::Value& buffer);

torch::lazy::NodePtr Sigmoid(const torch::lazy::Value& input);

torch::lazy::NodePtr SiLU(const torch::lazy::Value& input);

torch::lazy::NodePtr SiLUBackward(const torch::lazy::Value& grad_output,
                                  const torch::lazy::Value& input);

torch::lazy::NodePtr SigmoidBackward(const torch::lazy::Value& grad_output,
                                     const torch::lazy::Value& output);

torch::lazy::NodePtr LogSoftmaxBackwardOp(const torch::lazy::Value& grad_output,
                                          const torch::lazy::Value& output,
                                          int64_t dim);

torch::lazy::NodePtr SoftmaxBackwardOp(const torch::lazy::Value& grad_output,
                                       const torch::lazy::Value& output,
                                       int64_t dim);

torch::lazy::NodePtr Clamp(const torch::lazy::Value& input,
                           const torch::lazy::Value& min,
                           const torch::lazy::Value& max);

torch::lazy::NodePtr Ceil(const torch::lazy::Value& input);

torch::lazy::NodePtr Celu(const torch::lazy::Value& input,
                          const at::Scalar& alpha);

torch::lazy::NodePtr Round(const torch::lazy::Value& input);

torch::lazy::NodePtr Trunc(const torch::lazy::Value& input);

torch::lazy::NodePtr FracOp(const torch::lazy::Value& input);

torch::lazy::NodePtr Ger(const torch::lazy::Value& input,
                         const torch::lazy::Value& other);

torch::lazy::NodePtr AddMatMulOp(const torch::lazy::Value& input,
                                 const torch::lazy::Value& weight,
                                 const torch::lazy::Value& bias);

torch::lazy::NodePtr Dot(const torch::lazy::Value& input,
                         const torch::lazy::Value& weight);

torch::lazy::NodePtr MatMul(const torch::lazy::Value& lhs,
                            const torch::lazy::Value& rhs);

torch::lazy::NodePtr AdaptiveMaxPool2dBackward(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input);

torch::lazy::NodePtr AdaptiveAvgPool2dBackward(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input);

torch::lazy::NodePtr AdaptiveAvgPool3dBackward(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input);

torch::lazy::NodePtr ComparisonOp(c10::Symbol kind,
                                  const torch::lazy::Value& input,
                                  const torch::lazy::Value& other);

torch::lazy::NodePtr Where(const torch::lazy::Value& condition,
                           const torch::lazy::Value& input,
                           const torch::lazy::Value& other);

torch::lazy::NodePtr ARange(const at::Scalar& start, const at::Scalar& end,
                            const at::Scalar& step, at::ScalarType scalar_type);

torch::lazy::NodePtr BroadcastTensors(
    c10::ArrayRef<torch::lazy::Value> tensors);

torch::lazy::NodePtr Norm(const torch::lazy::Value& input,
                          const c10::optional<at::Scalar>& p,
                          c10::optional<at::ScalarType> dtype,
                          absl::Span<const int64_t> dims, bool keepdim);

torch::lazy::NodePtr Identity(int64_t lines, int64_t cols,
                              xla::PrimitiveType element_type);

torch::lazy::NodePtr Elu(const torch::lazy::Value& input,
                         const at::Scalar& alpha, const at::Scalar& scale,
                         const at::Scalar& input_scale);

torch::lazy::NodePtr EluBackward(const torch::lazy::Value& grad_output,
                                 const torch::lazy::Value& output,
                                 const at::Scalar& alpha,
                                 const at::Scalar& scale,
                                 const at::Scalar& input_scale);

torch::lazy::NodePtr Gelu(const torch::lazy::Value& input);

torch::lazy::NodePtr GeluBackward(const torch::lazy::Value& grad,
                                  const torch::lazy::Value& input);

torch::lazy::NodePtr Lshift(const torch::lazy::Value& input,
                            const at::Scalar& other);

torch::lazy::NodePtr Lshift(const torch::lazy::Value& input,
                            const torch::lazy::Value& other);

torch::lazy::NodePtr Rshift(const torch::lazy::Value& input,
                            const at::Scalar& other);

torch::lazy::NodePtr Rshift(const torch::lazy::Value& input,
                            const torch::lazy::Value& other);

torch::lazy::NodePtr Remainder(const torch::lazy::Value& input,
                               const torch::lazy::Value& divisor);

torch::lazy::NodePtr MaxUnary(const torch::lazy::Value& input);

torch::lazy::NodePtr MinUnary(const torch::lazy::Value& input);

torch::lazy::NodePtr Take(const torch::lazy::Value& input,
                          const torch::lazy::Value& index);

torch::lazy::NodePtr TanhGelu(const torch::lazy::Value& input);

torch::lazy::NodePtr TanhGeluBackward(const torch::lazy::Value& grad,
                                      const torch::lazy::Value& input);

torch::lazy::NodePtr IsNan(const torch::lazy::Value& input);

torch::lazy::NodePtr BaddBmm(const torch::lazy::Value& lhs,
                             const torch::lazy::Value& rhs,
                             const torch::lazy::Value& bias,
                             const torch::lazy::Value& product_multiplier,
                             const torch::lazy::Value& bias_multiplier);

torch::lazy::NodePtr Lerp(const torch::lazy::Value& start,
                          const torch::lazy::Value& end,
                          const torch::lazy::Value& weight);

torch::lazy::NodePtr LogicalNot(const torch::lazy::Value& input);

torch::lazy::NodePtr LogicalXor(const torch::lazy::Value& input,
                                const torch::lazy::Value& other);

torch::lazy::NodePtr LogicalAnd(const torch::lazy::Value& input,
                                const torch::lazy::Value& other);

torch::lazy::NodePtr LogicalOr(const torch::lazy::Value& input,
                               const torch::lazy::Value& other);

torch::lazy::NodePtr XLogY(const torch::lazy::Value& input,
                           const torch::lazy::Value& other);

torch::lazy::NodePtr NanToNum(const torch::lazy::Value& input,
                              const torch::lazy::Value& nan,
                              const torch::lazy::Value& posinf,
                              const torch::lazy::Value& neginf);

torch::lazy::NodePtr SLogDet(const torch::lazy::Value& input);

torch::lazy::NodePtr Softplus(const torch::lazy::Value& input,
                              const torch::lazy::Value& beta,
                              const torch::lazy::Value& threshold);

torch::lazy::NodePtr Selu(const torch::lazy::Value& input);

torch::lazy::NodePtr DynamicExpand(const XlaValue& input,
                                   const std::vector<XlaValue>& dimensions,
                                   const std::vector<int64_t> upper_bounds,
                                   const std::vector<bool> dynamic_dims);

}  // namespace torch_xla
