#pragma once

// This header can depend on ops/ and ir.h, as well as system/c++, tensorflow,
// PT,... but not on other PT/XLA headers.

#include <memory>

#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ops/constant.h"
#include "torch_xla/csrc/ops/cross_replica_sum.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/ops/generic.h"
#include "torch_xla/csrc/ops/scalar.h"

namespace torch_xla {
namespace ir {
namespace ops {

inline NodePtr ScalarOp(at::Scalar value, xla::Shape shape) {
  return MakeNode<Scalar>(value, std::move(shape));
}
inline NodePtr ScalarOp(at::Scalar value, xla::PrimitiveType type) {
  return MakeNode<Scalar>(value, type);
}

inline NodePtr ConstantOp(xla::Literal value) {
  return MakeNode<Constant>(std::move(value));
}

inline NodePtr DeviceDataOp(
    std::shared_ptr<xla::ComputationClient::Data> data) {
  return MakeNode<DeviceData>(std::move(data));
}

inline NodePtr GenericOp(OpKind op,
                         tensorflow::gtl::ArraySlice<const Value> operands,
                         xla::Shape shape, Generic::LowerFn lower_fn,
                         size_t num_outputs = 1) {
  return MakeNode<Generic>(std::move(op), operands, std::move(shape),
                           std::move(lower_fn), num_outputs);
}

inline NodePtr CrossReplicaSumOp(const Value& operand, double scale,
                                 std::vector<std::vector<xla::int64>> groups) {
  return MakeNode<CrossReplicaSum>(operand, scale, std::move(groups));
}

NodePtr Acos(const Value& input);

NodePtr Cos(const Value& input);

NodePtr Cosh(const Value& input);

NodePtr Asin(const Value& input);

NodePtr Sin(const Value& input);

NodePtr Sinh(const Value& input);

NodePtr Atan(const Value& input);

NodePtr Atan2(const Value& input, const Value& other);

NodePtr Tan(const Value& input);

NodePtr Tanh(const Value& input);

NodePtr Neg(const Value& input);

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

NodePtr Pow(const Value& input, const Value& exponent);

NodePtr Fmod(const Value& dividend, const Value& divisor);

NodePtr TransposeOp(const Value& input, xla::int64 dim0, xla::int64 dim1);

NodePtr Sigmoid(const Value& input);

NodePtr LogSoftmaxBackwardOp(const Value& grad_output, const Value& output,
                             xla::int64 dim);

NodePtr SoftmaxBackwardOp(const Value& grad_output, const Value& output,
                          xla::int64 dim);

NodePtr Clamp(const Value& input, c10::optional<at::Scalar> min,
              c10::optional<at::Scalar> max);

NodePtr Ceil(const Value& input);

NodePtr Floor(const Value& input);

NodePtr Trunc(const Value& input);

NodePtr FracOp(const Value& input);

NodePtr AddMatMulOp(const Value& input, const Value& weight, const Value& bias);

NodePtr Dot(const Value& input, const Value& weight);

NodePtr MatMul(const Value& lhs, const Value& rhs);

NodePtr NllLossOp(const Value& logits, const Value& labels);

NodePtr NllLossBackwardOp(const Value& logits, const Value& labels);

NodePtr AdaptiveAvgPool2dBackward(const Value& grad_output, const Value& input);

NodePtr ComparisonOp(c10::Symbol kind, const Value& input, const Value& other);

NodePtr ComparisonOp(c10::Symbol kind, const Value& input, at::Scalar other);

NodePtr Where(const Value& condition, const Value& input, const Value& other);

NodePtr ARange(at::Scalar start, at::Scalar end, at::Scalar step,
               at::ScalarType scalar_type);

NodePtr BroadcastTensors(tensorflow::gtl::ArraySlice<const Value> tensors);

NodePtr Norm(const Value& input, c10::optional<at::Scalar> p,
             c10::optional<at::ScalarType> dtype,
             tensorflow::gtl::ArraySlice<const xla::int64> dims, bool keepdim);

NodePtr Identity(xla::int64 lines, xla::int64 cols,
                 xla::PrimitiveType element_type);

NodePtr Elu(const Value& input, at::Scalar alpha, at::Scalar scale,
            at::Scalar input_scale);

NodePtr EluBackward(const Value& grad_output, const Value& output,
                    at::Scalar alpha, at::Scalar scale, at::Scalar input_scale);

NodePtr Lshift(const Value& input, at::Scalar other);

NodePtr Lshift(const Value& input, const Value& other);

NodePtr Rshift(const Value& input, at::Scalar other);

NodePtr Rshift(const Value& input, const Value& other);

NodePtr Remainder(const Value& input, const Value& divisor);

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
