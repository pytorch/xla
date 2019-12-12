#pragma once

#include <c10/core/Scalar.h>
#include <c10/util/Optional.h>

#include <functional>
#include <vector>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch_xla {

// Miscellaneous helpers for XLA lowering.
class XlaHelpers {
 public:
  struct MinMax {
    at::Scalar min;
    at::Scalar max;
  };

  template <class T>
  static xla::Literal ScalarLiteral(T scalar_value, xla::PrimitiveType type) {
    switch (type) {
      case xla::PrimitiveType::F64:
        return xla::LiteralUtil::CreateR0<double>(scalar_value);
      case xla::PrimitiveType::F32:
        return xla::LiteralUtil::CreateR0<float>(scalar_value);
      case xla::PrimitiveType::BF16:
        return xla::LiteralUtil::CreateR0<tensorflow::bfloat16>(
            static_cast<tensorflow::bfloat16>(scalar_value));
      case xla::PrimitiveType::S64:
        return xla::LiteralUtil::CreateR0<xla::int64>(scalar_value);
      case xla::PrimitiveType::S32:
        return xla::LiteralUtil::CreateR0<xla::int32>(scalar_value);
      case xla::PrimitiveType::S16:
        return xla::LiteralUtil::CreateR0<xla::int16>(scalar_value);
      case xla::PrimitiveType::S8:
        return xla::LiteralUtil::CreateR0<xla::int8>(scalar_value);
      case xla::PrimitiveType::U8:
        return xla::LiteralUtil::CreateR0<xla::uint8>(scalar_value);
      case xla::PrimitiveType::PRED:
        return xla::LiteralUtil::CreateR0<bool>(scalar_value);
      default:
        return xla::LiteralUtil::CreateR0<T>(scalar_value);
    }
  }

  // Creates a XLA constant for the given scalar_value.
  template <class T>
  static xla::XlaOp ScalarValue(T scalar_value, xla::XlaBuilder* builder) {
    xla::Literal scalar_literal = xla::LiteralUtil::CreateR0<T>(scalar_value);
    return xla::ConstantLiteral(builder, scalar_literal);
  }

  template <class T>
  static xla::XlaOp ScalarValue(T scalar_value, xla::PrimitiveType type,
                                xla::XlaBuilder* builder) {
    return xla::ConstantLiteral(builder, ScalarLiteral(scalar_value, type));
  }

  static xla::XlaOp ScalarValue(at::Scalar scalar_value,
                                xla::PrimitiveType type,
                                xla::XlaBuilder* builder) {
    if (scalar_value.isFloatingPoint()) {
      return ScalarValue(scalar_value.toDouble(), type, builder);
    }
    XLA_CHECK(scalar_value.isIntegral()) << "Scalar type not supported";
    return ScalarValue(static_cast<xla::int64>(scalar_value.toLong()), type,
                       builder);
  }

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

  static std::vector<xla::int64> GetAllDimensions(size_t rank) {
    return xla::util::Iota<xla::int64>(rank);
  }

  static std::vector<xla::int64> GetAllDimensions(const xla::Shape& shape) {
    return xla::util::Iota<xla::int64>(shape.rank());
  }

  static xla::XlaOp CreateReturnValue(xla::XlaBuilder* builder,
                                      const std::vector<xla::XlaOp>& outputs);

  // Creates a scalar broadcasted to a given shape.
  template <class T>
  static xla::XlaOp ScalarBroadcast(
      T scalar_value, xla::PrimitiveType type,
      tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
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

  // Creates a convolution or dot precision configuration.
  static xla::PrecisionConfig BuildPrecisionConfig(
      const xla::PrecisionConfig::Precision conv_precision);

  // Converts an iterable container to a vector XLA int64's.
  template <typename S>
  static std::vector<xla::int64> I64List(const S& input) {
    return xla::util::ToVector<xla::int64>(input);
  }

  static c10::optional<xla::int64> I64Optional(c10::optional<int64_t> opt) {
    return opt ? c10::optional<xla::int64>(*opt) : c10::nullopt;
  }

  // Creates an XLA padding configuration from a n-dimensional padding list.
  static xla::PaddingConfig MakeXlaPaddingConfigFromNdPadding(
      tensorflow::gtl::ArraySlice<const xla::int64> padding);

  // Creates a set of dimension by dropping the drop_dims ones.
  static std::vector<xla::int64> DropDimensions(
      tensorflow::gtl::ArraySlice<const xla::int64> sizes,
      tensorflow::gtl::ArraySlice<const xla::int64> drop_dims);

  // Get the canonical dimension index in the [0, rank) interval. Negative
  // indices are interpreted as follows: -1 is rank-1, -2 is rank-2 etc.
  static xla::int64 GetCanonicalDimensionIndex(xla::int64 dim, xla::int64 rank);

  // Same as above, for multiple dimensions.
  static std::vector<xla::int64> GetCanonicalDimensionIndices(
      tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
      xla::int64 rank);

  // Returns the canonical position in the dim dimension, handling negative
  // values for the position.
  static xla::int64 GetCanonicalPosition(
      tensorflow::gtl::ArraySlice<const xla::int64> dimensions, xla::int64 dim,
      xla::int64 pos);

  // Retrieves the dynamic dimension of an input shape, or returns -1 if none.
  static xla::int64 GetDynamicDimension(const xla::Shape& shape);

  static xla::XlaOp GetDimensionsSize(
      tensorflow::gtl::ArraySlice<const xla::XlaOp> inputs,
      tensorflow::gtl::ArraySlice<const xla::int64> dimensions);

  // Retrieves type's minimum and maximum values.
  static MinMax MinMaxValues(xla::PrimitiveType type);

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

  // Gathers the input using the order specified by the permutation. For each i,
  // output[i] = input[permutation[i]]. The given permutation must be the same
  // size as the input.
  template <typename Container>
  static std::vector<typename Container::value_type> Permute(
      tensorflow::gtl::ArraySlice<const xla::int64> permutation,
      const Container& input) {
    using T = typename Container::value_type;
    XLA_CHECK(xla::IsPermutation(permutation, input.size()))
        << "Invalid permutation specified";
    std::vector<T> output(input.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
      output[i] = input[permutation[i]];
    }
    return output;
  }

  // Creates a transposition from the given input and dimensions.
  static std::vector<xla::int64> MakeTransposePermutation(xla::int64 dim0,
                                                          xla::int64 dim1,
                                                          xla::int64 rank);

  // Performs type promotion to make sure both operations return the same type.
  static std::pair<xla::XlaOp, xla::XlaOp> PromoteValues(xla::XlaOp op1,
                                                         xla::XlaOp op2);

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

  // Calculates the protomoted shape to which the input shapes should be
  // broadcasted for an elementwise operation. The size of the common dimensions
  // (2,3,4 for shape1, and 0,1,2 for shape2) must either match, or either one
  // of the two be 1.
  // Example:
  //   shape1       = [9, 7, 6, 1, 2]
  //   shape2       =       [6, 5, 2]
  //   result_shape = [9, 7, 6, 5, 2]
  static std::vector<xla::int64> GetPromotedShape(
      tensorflow::gtl::ArraySlice<const xla::int64> shape1_dims,
      tensorflow::gtl::ArraySlice<const xla::int64> shape2_dims);

  static xla::Shape GetPromotedShape(const xla::Shape& shape1,
                                     const xla::Shape& shape2);

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

  template <typename T>
  static xla::Literal Range(T start, T end, T step) {
    return xla::LiteralUtil::CreateR1<T>(xla::util::Range<T>(start, end, step));
  }

  static xla::PrecisionConfig::Precision mat_mul_precision() {
    return s_mat_mul_precision;
  }

  static void set_mat_mul_precision(xla::PrecisionConfig::Precision precision) {
    s_mat_mul_precision = precision;
  }

 private:
  static xla::PrecisionConfig::Precision s_mat_mul_precision;
};

}  // namespace torch_xla
