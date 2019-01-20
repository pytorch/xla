#pragma once

#include <functional>
#include <vector>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch/csrc/jit/ir.h"

namespace torch_xla {

// Miscellaneous helpers for XLA lowering.
class XlaHelpers {
 public:
  template <class T>
  static xla::Literal ScalarLiteral(T scalar_value, xla::PrimitiveType type) {
    switch (type) {
      case xla::PrimitiveType::F32:
        return xla::LiteralUtil::CreateR0<float>(scalar_value);
      case xla::PrimitiveType::BF16:
        return xla::LiteralUtil::CreateR0<tensorflow::bfloat16>(
            static_cast<tensorflow::bfloat16>(scalar_value));
      case xla::PrimitiveType::S64:
        return xla::LiteralUtil::CreateR0<xla::int64>(scalar_value);
      default:
        return xla::LiteralUtil::CreateR0<T>(scalar_value);
    }
  }

  // Creates a XLA constant for the given scalar_value.
  template <class T>
  static xla::XlaOp ScalarValue(T scalar_value, xla::XlaBuilder* builder) {
    const auto scalar_literal = xla::LiteralUtil::CreateR0<T>(scalar_value);
    return xla::ConstantLiteral(builder, scalar_literal);
  }

  template <class T>
  static xla::XlaOp ScalarValue(T scalar_value, xla::PrimitiveType type,
                                xla::XlaBuilder* builder) {
    return xla::ConstantLiteral(builder, ScalarLiteral(scalar_value, type));
  }

  // Returns the list of dimension sizes for the given shape.
  static std::vector<xla::int64> ShapeSizes(const xla::Shape& shape);

  // Returns the shape of the given XLA operation.
  static xla::Shape ShapeOfXlaOp(const xla::XlaOp& op);

  // Returns the list of dimension sizes for the given XLA operation.
  static std::vector<xla::int64> SizesOfXlaOp(const xla::XlaOp& op);

  // Returns the value type of given XLA operation.
  static xla::PrimitiveType TypeOfXlaOp(const xla::XlaOp& op);

  static xla::XlaOp CreateReturnValue(xla::XlaBuilder* builder,
                                      const std::vector<xla::XlaOp>& outputs);

  // Creates a scalar broadcasted to a given shape.
  template <class T>
  static xla::XlaOp ScalarBroadcast(T scalar_value, const xla::Shape& shape,
                                    xla::XlaBuilder* builder) {
    auto scalar_op =
        ScalarValue<T>(scalar_value, shape.element_type(), builder);
    return xla::Broadcast(scalar_op, ShapeSizes(shape));
  }

  // Creates a convolution or dot precision configuration.
  static xla::PrecisionConfig BuildPrecisionConfig(
      const xla::PrecisionConfig::Precision conv_precision);

  // Converts int64_t's to XLA int64's.
  static std::vector<xla::int64> I64List(const at::IntList& input);

  // Creates an XLA padding configuration from a padding attribute value.
  static xla::PaddingConfig MakeXlaPaddingConfig(
      tensorflow::gtl::ArraySlice<const xla::int64> padding);

  // Creates a binary add computation.
  static xla::XlaComputation CreateAddComputation(xla::PrimitiveType type);

  // Converts the given scalar type to an XLA primitive type.
  static xla::PrimitiveType MakeXlaPrimitiveType(at::ScalarType scalar_type);

  // Performs type promotion to make sure both operations return the same type.
  static std::pair<xla::XlaOp, xla::XlaOp> PromoteValues(const xla::XlaOp& op1,
                                                         const xla::XlaOp& op2);

  // Performs type promotion, by casting the second operation to the type of the
  // first, if different.
  static std::pair<xla::XlaOp, xla::XlaOp> PromoteSecondValue(
      const xla::XlaOp& op1, const xla::XlaOp& op2);

  // Eventually performs a broadcast to make sure the shapes of the returned
  // xla::XlaOp values have the same shape. The first returned xla::XlaOp is op1
  // or a broadcast of it, and the second returned xla::XlaOp is either op2 or a
  // broadcast ot it.
  static std::pair<xla::XlaOp, xla::XlaOp> PromoteShapes(const xla::XlaOp& op1,
                                                         const xla::XlaOp& op2);

  // Combines PromoteValues() and PromoteShapes() returning two operations which
  // match in shape and types.
  static std::pair<xla::XlaOp, xla::XlaOp> Promote(const xla::XlaOp& op1,
                                                   const xla::XlaOp& op2);

  // Combines PromoteSecondValue() and PromoteShapes() returning two operations
  // which match in shape and types.
  static std::pair<xla::XlaOp, xla::XlaOp> PromoteSecond(const xla::XlaOp& op1,
                                                         const xla::XlaOp& op2);

  // Calculates the protomoted shape to which the input shapes should be
  // broadcasted for an elementwise operation. The size of the common dimensions
  // (2,3,4 for shape1, and 0,1,2 for shape2) must either match, or either one
  // of the two be 1.
  // Example:
  //   shape1       = [9, 7, 6, 1, 2]
  //   shape2       =       [6, 5, 2]
  //   result_shape = [9, 7, 6, 5, 2]
  static xla::Shape GetPromotedShape(const xla::Shape& shape1,
                                     const xla::Shape& shape2);

  // Returns a new operations which broadcast the input operation into the
  // shape. The op_shape is the shape of the op operation, while shape should be
  // one that op is broadcast-able to (usually the result of a
  // GetPromotedShape() call). If op_shape matches shape, the op itself is
  // returned.
  static xla::XlaOp ImplicitBroadcast(const xla::XlaOp& op,
                                      const xla::Shape& op_shape,
                                      const xla::Shape& shape);

  // Performs the bin_op binary operation by promoting types and shapes of the
  // two input operands.
  static xla::XlaOp PromotedBinaryOp(
      const xla::XlaOp& op1, const xla::XlaOp& op2,
      const std::function<xla::XlaOp(const xla::XlaOp&, const xla::XlaOp&)>&
          bin_op);

  // Basic promoted binary operation implementation follow.
  static xla::XlaOp PromotedAdd(const xla::XlaOp& op1, const xla::XlaOp& op2) {
    return PromotedBinaryOp(
        op1, op2,
        [](const xla::XlaOp& op1, const xla::XlaOp& op2) { return op1 + op2; });
  }

  static xla::XlaOp PromotedSub(const xla::XlaOp& op1, const xla::XlaOp& op2) {
    return PromotedBinaryOp(
        op1, op2,
        [](const xla::XlaOp& op1, const xla::XlaOp& op2) { return op1 - op2; });
  }

  static xla::XlaOp PromotedMul(const xla::XlaOp& op1, const xla::XlaOp& op2) {
    return PromotedBinaryOp(
        op1, op2,
        [](const xla::XlaOp& op1, const xla::XlaOp& op2) { return op1 * op2; });
  }

  static xla::XlaOp PromotedDiv(const xla::XlaOp& op1, const xla::XlaOp& op2) {
    return PromotedBinaryOp(
        op1, op2,
        [](const xla::XlaOp& op1, const xla::XlaOp& op2) { return op1 / op2; });
  }

  // Checks whether BF16 should be used as default floating point type for XLA
  // computations.
  static bool UseBF16();
};

}  // namespace torch_xla
