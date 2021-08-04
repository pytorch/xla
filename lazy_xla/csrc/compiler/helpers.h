#pragma once

#include <c10/core/Scalar.h>
#include <c10/util/Optional.h>

#include <functional>
#include <tuple>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensors/computation_client/util.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"

namespace torch_lazy_tensors {
namespace compiler {

// Miscellaneous helpers for XLA lowering.
class XlaHelpers {
 public:
  struct DynamicSize {
    xla::XlaOp size;
    absl::optional<xla::int64> scalar_size;
  };

  // Creates a XLA constant for the given scalar_value.
  template <class T>
  static xla::XlaOp ScalarValue(T scalar_value, xla::XlaBuilder* builder) {
    xla::Literal scalar_literal = xla::LiteralUtil::CreateR0<T>(scalar_value);
    return xla::ConstantLiteral(builder, scalar_literal);
  }

  template <class T>
  static xla::XlaOp ScalarValue(T scalar_value, xla::PrimitiveType type,
                                xla::XlaBuilder* builder) {
    lazy_tensors::Literal literal = torch_lazy_tensors::Helpers::ScalarLiteral(
        scalar_value, LazyTensorPrimitiveType(type));
    LTC_CHECK_EQ(literal.shape().rank(), 0);
    xla::Literal xla_literal(XlaShape(literal.shape()));
    xla_literal.Set({}, literal.data<T>()[0]);
    return xla::ConstantLiteral(builder, xla_literal);
  }

  static xla::XlaOp ScalarValue(at::Scalar scalar_value,
                                xla::PrimitiveType type,
                                xla::XlaBuilder* builder) {
    if (scalar_value.isFloatingPoint()) {
      return ScalarValue(scalar_value.toDouble(), type, builder);
    }
    LTC_CHECK(scalar_value.isIntegral()) << "Scalar type not supported";
    return ScalarValue(static_cast<xla::int64>(scalar_value.toLong()), type,
                       builder);
  }

  static bool SameStaticDimensions(const xla::Shape& shape1,
                                   const xla::Shape& shape2);

  static xla::PrecisionConfig BuildPrecisionConfig(
      xla::PrecisionConfig::Precision conv_precision, int num_arguments = 2);

  static xla::Shape XlaShape(const lazy_tensors::Shape& shape);

  static lazy_tensors::Shape LazyTensorsShape(const xla::Shape& shape);

  // Performa a linear interpolation between value0 and value1, by calculating:
  //   result = value0 * alpha + value1 * (1 - alpha)
  static xla::XlaOp LinearInterpolation(xla::XlaOp value0, xla::XlaOp value1,
                                        double alpha);

  // Returns the shape of the given XLA operation.
  static const xla::Shape& ShapeOfXlaOp(xla::XlaOp op);

  // Returns the list of dimension sizes for the given XLA operation.
  static std::vector<xla::int64> SizesOfXlaOp(xla::XlaOp op);

  // Returns the value type of given XLA operation.
  static xla::PrimitiveType TypeOfXlaOp(xla::XlaOp op);

  static std::vector<xla::int64> GetAllDimensions(const xla::Shape& shape) {
    return xla::util::Iota<xla::int64>(shape.rank());
  }

  static xla::XlaOp BroadcastDimensions(xla::XlaOp input,
                                        absl::Span<const xla::int64> dimensions,
                                        absl::Span<const xla::int64> sizes);

  static xla::XlaOp CreateReturnValue(xla::XlaBuilder* builder,
                                      const std::vector<xla::XlaOp>& outputs);

  // Creates a scalar broadcasted to a given shape.
  template <class T>
  static xla::XlaOp ScalarBroadcast(T scalar_value, xla::PrimitiveType type,
                                    absl::Span<const xla::int64> dimensions,
                                    xla::XlaBuilder* builder) {
    xla::XlaOp scalar_op = ScalarValue<T>(scalar_value, type, builder);
    return xla::Broadcast(scalar_op, dimensions);
  }

  template <class T>
  static xla::XlaOp ScalarBroadcast(T scalar_value, const xla::Shape& shape,
                                    xla::XlaBuilder* builder) {
    return ScalarBroadcast<T>(scalar_value, shape.element_type(),
                              shape.dimensions(), builder);
  }

  static xla::XlaOp DynamicReshape(xla::XlaOp input,
                                   absl::Span<const xla::int64> output_sizes);

  static xla::XlaOp DynamicReshapeAs(xla::XlaOp input, const xla::Shape& shape);

  static DynamicSize GetDimensionsSize(absl::Span<const xla::XlaOp> inputs,
                                       absl::Span<const xla::int64> dimensions);

  // Creates a binary add computation.
  static xla::XlaComputation CreateAddComputation(xla::PrimitiveType type);

  // Creates a binary mul computation.
  static xla::XlaComputation CreateMulComputation(xla::PrimitiveType type);

  static xla::XlaComputation CreateMaxComputation(xla::PrimitiveType type);

  static xla::XlaComputation CreateMinComputation(xla::PrimitiveType type);

  static xla::XlaComputation CreateAndComputation(xla::PrimitiveType type);

  static xla::XlaComputation CreateOrComputation(xla::PrimitiveType type);

  // Returns an XLA operation which is a reshape to the expected rank, by
  // appending 1s to the major dimension. If offset is greater than zero, 1s
  // will be prepened to the minor dimension as well.
  // Expected condition: rank(input) + offset <= expected_rank
  static xla::XlaOp ReshapeToRank(xla::XlaOp input, xla::int64 expected_rank,
                                  xla::int64 offset = 0);

  static xla::XlaOp Flatten(xla::XlaOp input,
                            xla::Shape* input_shape = nullptr);

  static xla::XlaOp FlattenDimRange(xla::XlaOp input, xla::int64 start,
                                    xla::int64 range,
                                    xla::Shape* input_shape = nullptr);

  // Performs type promotion to make sure both operations return the same type.
  static std::pair<xla::XlaOp, xla::XlaOp> PromoteValues(xla::XlaOp op1,
                                                         xla::XlaOp op2);

  static std::tuple<xla::XlaOp, xla::XlaOp, xla::XlaOp> PromoteValues(
      xla::XlaOp op1, xla::XlaOp op2, xla::XlaOp op3);

  // Performs type promotion, by casting the second operation to the type of the
  // first, if different.
  static std::pair<xla::XlaOp, xla::XlaOp> PromoteSecondValue(xla::XlaOp op1,
                                                              xla::XlaOp op2);

  // Eventually performs a broadcast to make sure the shapes of the returned
  // xla::XlaOp values have the same shape. The first returned xla::XlaOp is op1
  // or a broadcast of it, and the second returned xla::XlaOp is either op2 or a
  // broadcast ot it.
  static std::pair<xla::XlaOp, xla::XlaOp> PromoteShapes(xla::XlaOp op1,
                                                         xla::XlaOp op2);

  // Combines PromoteValues() and PromoteShapes() returning two operations which
  // match in shape and types.
  static std::pair<xla::XlaOp, xla::XlaOp> Promote(xla::XlaOp op1,
                                                   xla::XlaOp op2);

  // Combines PromoteSecondValue() and PromoteShapes() returning two operations
  // which match in shape and types.
  static std::pair<xla::XlaOp, xla::XlaOp> PromoteSecond(xla::XlaOp op1,
                                                         xla::XlaOp op2);

  // Returns a new operations which broadcast the input operation into the
  // shape. The op_shape is the shape of the op operation, while shape should be
  // one that op is broadcast-able to (usually the result of a
  // GetPromotedShape() call). If op_shape matches shape, the op itself is
  // returned.
  static xla::XlaOp ImplicitBroadcast(xla::XlaOp op, const xla::Shape& op_shape,
                                      const xla::Shape& shape);

  // Performs the bin_op binary operation by promoting types and shapes of the
  // two input operands.
  static xla::XlaOp PromotedBinaryOp(
      xla::XlaOp op1, xla::XlaOp op2,
      const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& bin_op);

  // Basic promoted binary operation implementation follow.
  static xla::XlaOp PromotedAdd(xla::XlaOp op1, xla::XlaOp op2) {
    return PromotedBinaryOp(
        op1, op2, [](xla::XlaOp op1, xla::XlaOp op2) { return op1 + op2; });
  }

  static xla::XlaOp PromotedSub(xla::XlaOp op1, xla::XlaOp op2) {
    return PromotedBinaryOp(
        op1, op2, [](xla::XlaOp op1, xla::XlaOp op2) { return op1 - op2; });
  }

  static xla::XlaOp PromotedMul(xla::XlaOp op1, xla::XlaOp op2) {
    return PromotedBinaryOp(
        op1, op2, [](xla::XlaOp op1, xla::XlaOp op2) { return op1 * op2; });
  }

  static xla::XlaOp PromotedDiv(xla::XlaOp op1, xla::XlaOp op2) {
    return PromotedBinaryOp(
        op1, op2, [](xla::XlaOp op1, xla::XlaOp op2) { return op1 / op2; });
  }

  static xla::XlaOp PromotedLogicalBinaryOp(
      xla::XlaOp op1, xla::XlaOp op2,
      const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& bin_op);

  // Creates an XLA padding configuration from a n-dimensional padding list.
  static xla::PaddingConfig MakeXlaPaddingConfigFromNdPadding(
      absl::Span<const xla::int64> padding);

  static lazy_tensors::PrimitiveType LazyTensorPrimitiveType(
      xla::PrimitiveType type);

  static xla::PrecisionConfig::Precision mat_mul_precision() {
    return s_mat_mul_precision;
  }

  static void set_mat_mul_precision(xla::PrecisionConfig::Precision precision) {
    s_mat_mul_precision = precision;
  }

 private:
  static xla::PrecisionConfig::Precision s_mat_mul_precision;
};

}  // namespace compiler
}  // namespace torch_lazy_tensors
