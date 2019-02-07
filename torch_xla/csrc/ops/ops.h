#pragma once

// This header can depend on ops/ and ir.h, as well as system/c++, tensorflow,
// PT,... but not on other PT/XLA headers.

#include <memory>

#include "ir.h"
#include "ops/constant.h"
#include "ops/cross_replica_sum.h"
#include "ops/device_data.h"
#include "ops/generic.h"
#include "ops/scalar.h"

namespace torch_xla {
namespace ir {
namespace ops {

inline NodePtr ScalarOp(double value, xla::Shape shape) {
  return std::make_shared<Scalar>(value, std::move(shape));
}
inline NodePtr ScalarOp(double value, xla::PrimitiveType type) {
  return std::make_shared<Scalar>(value, type);
}

inline NodePtr ConstantOp(xla::Literal value) {
  return std::make_shared<Constant>(std::move(value));
}

inline NodePtr DeviceDataOp(
    std::shared_ptr<xla::ComputationClient::Data> data) {
  return std::make_shared<DeviceData>(std::move(data));
}

inline NodePtr GenericOp(OpKind op,
                         tensorflow::gtl::ArraySlice<const Value> operands,
                         xla::Shape shape, Generic::LowerFn lower_fn,
                         size_t num_outputs = 1) {
  return std::make_shared<Generic>(std::move(op), operands, std::move(shape),
                                   std::move(lower_fn), num_outputs);
}

inline NodePtr CrossReplicaSumOp(const Value& operand,
                                 std::vector<std::vector<xla::int64>> groups) {
  return std::make_shared<CrossReplicaSum>(operand, std::move(groups));
}

NodePtr ReluOp(const Value& input);

NodePtr TransposeOp(const Value& input);

NodePtr AddMatMulOp(const Value& input, const Value& weight, const Value& bias,
                    bool use_full_conv_precision);

NodePtr MatMulOp(const Value& input, const Value& weight,
                 bool use_full_conv_precision);

NodePtr NllLossOp(const Value& logits, const Value& labels);

NodePtr NllLossBackwardOp(const Value& logits, const Value& labels);

// Placeholder node which is never to be used. Using it would throw an error
// during lowering.
NodePtr NotSupportedOp(c10::Symbol node_symbol, xla::Shape shape);

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
