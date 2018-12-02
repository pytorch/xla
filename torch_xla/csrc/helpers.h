#pragma once

#include <vector>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

// Miscellaneous helpers for XLA lowering.
class XlaHelpers {
 public:
  // Creates a XLA constant for the given scalar_value.
  template <class T>
  static xla::XlaOp ScalarValue(T scalar_value, xla::XlaBuilder* builder) {
    const auto scalar_literal = xla::LiteralUtil::CreateR0<T>(scalar_value);
    return xla::ConstantLiteral(builder, scalar_literal);
  }

  template <class T>
  static xla::XlaOp ScalarValue(T scalar_value, xla::PrimitiveType type,
                                xla::XlaBuilder* builder) {
    xla::Literal scalar_literal;
    switch (type) {
      case xla::PrimitiveType::F32:
        scalar_literal = xla::LiteralUtil::CreateR0<float>(scalar_value);
        break;
      case xla::PrimitiveType::BF16:
        scalar_literal = xla::LiteralUtil::CreateR0<tensorflow::bfloat16>(
            static_cast<tensorflow::bfloat16>(scalar_value));
        break;
      case xla::PrimitiveType::S64:
        scalar_literal = xla::LiteralUtil::CreateR0<xla::int64>(scalar_value);
        break;
      default:
        scalar_literal = xla::LiteralUtil::CreateR0<T>(scalar_value);
    }
    return xla::ConstantLiteral(builder, scalar_literal);
  }

  // Returns the list of dimension sizes for the given shape.
  static std::vector<xla::int64> ShapeSizes(const xla::Shape& shape);

  // Returns the shape of the given XLA operation.
  static xla::Shape ShapeOfXlaOp(const xla::XlaOp& op);

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

  // Returns the dimension sizes for the given tensor.
  static std::vector<int64_t> TensorDimensionSizes(const Value* tensor);

  // Converts int64_t's to XLA int64's.
  static std::vector<xla::int64> I64List(const at::IntList& input);

  // Creates an XLA padding configuration from a padding attribute value.
  static xla::PaddingConfig MakeXlaPaddingConfig(
      const std::vector<int64_t>& padding);

  // Creates a binary add computation.
  static xla::XlaComputation CreateAddComputation(xla::PrimitiveType type);

  // Converts the given scalar type to an XLA primitive type.
  static xla::PrimitiveType MakeXlaPrimitiveType(
      const at::ScalarType scalar_type);

  // Performs type promotion to make sure both operations return the same type.
  static std::pair<xla::XlaOp, xla::XlaOp> PromoteValues(const xla::XlaOp& op1,
                                                         const xla::XlaOp& op2);

  // Checks whether BF16 should be used as default floating point type for XLA
  // computations.
  static bool UseBF16();
};

}  // namespace jit
}  // namespace torch
