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

inline NodePtr CrossReplicaSumOp(const Value& operand,
                                 std::vector<std::vector<xla::int64>> groups) {
  return MakeNode<CrossReplicaSum>(operand, std::move(groups));
}

NodePtr Acos(const Value& input);

NodePtr Cos(const Value& input);

NodePtr Cosh(const Value& input);

NodePtr Asin(const Value& input);

NodePtr Sin(const Value& input);

NodePtr Sinh(const Value& input);

NodePtr Neg(const Value& input);

NodePtr Abs(const Value& input);

NodePtr ReluOp(const Value& input);

NodePtr TransposeOp(const Value& input);

NodePtr Min(const Value& input, const Value& other);

NodePtr Max(const Value& input, const Value& other);

NodePtr Exp(const Value& input);

NodePtr Expm1(const Value& input);

NodePtr Erf(const Value& input);

NodePtr Erfc(const Value& input);

NodePtr Log(const Value& input);

NodePtr Log1p(const Value& input);

NodePtr Sqrt(const Value& input);

NodePtr Pow(const Value& input, const Value& exponent);

NodePtr Clamp(const Value& input, c10::optional<at::Scalar> min,
              c10::optional<at::Scalar> max);

NodePtr AddMatMulOp(const Value& input, const Value& weight, const Value& bias,
                    bool use_full_conv_precision);

NodePtr MatMulOp(const Value& input, const Value& weight,
                 bool use_full_conv_precision);

NodePtr NllLossOp(const Value& logits, const Value& labels);

NodePtr NllLossBackwardOp(const Value& logits, const Value& labels);

NodePtr AdaptiveAvgPool2dBackward(const Value& grad_output, const Value& input);

NodePtr ComparisonOp(c10::Symbol kind, const Value& input, const Value& other);

NodePtr ComparisonOp(c10::Symbol kind, const Value& input,
                     const at::Scalar& other);

NodePtr Where(const Value& condition, const Value& input, const Value& other);

// Placeholder node which is never to be used. Using it would throw an error
// during lowering.
NodePtr NotSupportedOp(c10::Symbol node_symbol, xla::Shape shape);

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
