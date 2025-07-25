#ifndef XLA_TORCH_XLA_CSRC_HELPERS_H_
#define XLA_TORCH_XLA_CSRC_HELPERS_H_

#include <c10/core/Scalar.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/core/util.h>

#include <functional>
#include <optional>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/runtime/util.h"
#include "tsl/platform/bfloat16.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal_util.h"
#include "xla/permutation_util.h"
#include "xla/types.h"

namespace torch_xla {

// Miscellaneous helpers for XLA lowering.
class XlaHelpers {
 public:
  struct MinMax {
    at::Scalar min;
    at::Scalar max;
  };

  struct DynamicSize {
    xla::XlaOp size;
    absl::optional<int64_t> scalar_size;
  };

  struct DynamicReshapeInfo {
    xla::Shape output_shape;
    int64_t dynamic_dimension = -1;
  };

  template <class T>
  static xla::Literal ScalarLiteral(T scalar_value, xla::PrimitiveType type) {
    switch (type) {
      case xla::PrimitiveType::F64:
        return xla::LiteralUtil::CreateR0<double>(scalar_value);
      case xla::PrimitiveType::F32:
        return xla::LiteralUtil::CreateR0<float>(scalar_value);
      case xla::PrimitiveType::BF16:
        return xla::LiteralUtil::CreateR0<tsl::bfloat16>(
            static_cast<tsl::bfloat16>(static_cast<float>(scalar_value)));
      case xla::PrimitiveType::F16:
        return xla::LiteralUtil::CreateR0<xla::half>(
            static_cast<xla::half>(static_cast<float>(scalar_value)));
      case xla::PrimitiveType::S64:
        return xla::LiteralUtil::CreateR0<int64_t>(scalar_value);
      case xla::PrimitiveType::U64:
        return xla::LiteralUtil::CreateR0<uint64_t>(scalar_value);
      case xla::PrimitiveType::S32:
        return xla::LiteralUtil::CreateR0<int32_t>(scalar_value);
      case xla::PrimitiveType::U32:
        return xla::LiteralUtil::CreateR0<uint32_t>(scalar_value);
      case xla::PrimitiveType::S16:
        return xla::LiteralUtil::CreateR0<int16_t>(scalar_value);
      case xla::PrimitiveType::U16:
        return xla::LiteralUtil::CreateR0<uint16_t>(scalar_value);
      case xla::PrimitiveType::S8:
        return xla::LiteralUtil::CreateR0<int8_t>(scalar_value);
      case xla::PrimitiveType::U8:
        return xla::LiteralUtil::CreateR0<uint8_t>(scalar_value);
      case xla::PrimitiveType::PRED:
        return xla::LiteralUtil::CreateR0<bool>(scalar_value);
      case xla::PrimitiveType::C64:
        return xla::LiteralUtil::CreateR0<xla::complex64>(scalar_value);
      case xla::PrimitiveType::C128:
        return xla::LiteralUtil::CreateR0<xla::complex128>(scalar_value);
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

  static xla::XlaOp ScalarValue(const at::Scalar& scalar_value,
                                xla::PrimitiveType type,
                                xla::XlaBuilder* builder) {
    if (scalar_value.isFloatingPoint()) {
      return ScalarValue(scalar_value.toDouble(), type, builder);
    }
    if (scalar_value.isBoolean()) {
      return ScalarValue(scalar_value.toBool(), type, builder);
    }
    XLA_CHECK(scalar_value.isIntegral()) << "Scalar type not supported";
    return ScalarValue(static_cast<int64_t>(scalar_value.toLong()), type,
                       builder);
  }

  // Performa a linear interpolation between value0 and value1, by calculating:
  //   result = value0 * alpha + value1 * (1 - alpha)
  static xla::XlaOp LinearInterpolation(xla::XlaOp value0, xla::XlaOp value1,
                                        double alpha);

  // Returns the list of dimension sizes for the given XLA operation.
  static std::vector<int64_t> SizesOfXlaOp(xla::XlaOp op);

  // Returns the value type of given XLA operation.
  static xla::PrimitiveType TypeOfXlaOp(xla::XlaOp op);

  static std::vector<int64_t> GetAllDimensions(size_t rank) {
    return torch::lazy::Iota<int64_t>(rank);
  }

  static std::vector<int64_t> GetAllDimensions(const xla::Shape& shape) {
    return torch::lazy::Iota<int64_t>(shape.dimensions_size());
  }

  static xla::XlaOp BroadcastDimensions(xla::XlaOp input,
                                        absl::Span<const int64_t> dimensions,
                                        absl::Span<const int64_t> sizes);

  static xla::XlaOp CreateReturnValue(xla::XlaBuilder* builder,
                                      const std::vector<xla::XlaOp>& outputs);

  // Creates a scalar broadcasted to a given shape.
  template <class T>
  static xla::XlaOp ScalarBroadcast(T scalar_value, xla::PrimitiveType type,
                                    absl::Span<const int64_t> dimensions,
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

  static absl::optional<DynamicReshapeInfo> GetDynamicReshapeInfo(
      const xla::Shape& input_shape, absl::Span<const int64_t> output_sizes);

  static xla::Shape GetDynamicReshape(const xla::Shape& input_shape,
                                      absl::Span<const int64_t> output_sizes);

  static xla::XlaOp DynamicReshape(xla::XlaOp input,
                                   absl::Span<const int64_t> output_sizes);

  static bool IsUnboundedDynamic(const xla::Shape& shape);

  static bool IsUnboundedDynamismEnabled() {
    return runtime::sys_util::GetEnvBool("EXPERIMENTAL_XLA_UNBOUNDED_DYNAMISM",
                                         false);
  }

  // Creates custom_call to express dynamic reshape op using the dimension
  // sizes of 'aux_input'.
  static xla::XlaOp DynamicUnboundedReshape(
      xla::XlaOp input, xla::XlaOp aux_input,
      absl::Span<const int64_t> output_sizes);

  // Broadcasts 'input' shape to
  // shape(aux_input)[aux_input_dimensions] x shape(input).
  // This method is used as a replacement for xla::Broadcast when unbounded
  // dynamic shapes are involved.
  static xla::XlaOp DynamicUnboundedBroadcast(
      xla::XlaOp input, xla::XlaOp aux_input,
      const std::vector<int64_t>& aux_input_dimensions);

  static xla::XlaOp DynamicBroadcastInDim(
      xla::XlaOp op, const xla::Shape& final_shape,
      xla::XlaOp final_broadcast_dimensions);

  static xla::XlaOp DynamicReshapeAs(xla::XlaOp input, const xla::Shape& shape);

  static bool SameStaticDimensions(const xla::Shape& shape1,
                                   const xla::Shape& shape2);

  static xla::PrecisionConfig BuildPrecisionConfig(
      xla::PrecisionConfig::Precision conv_precision, int num_arguments = 2);

  // Converts an iterable container to a vector XLA int64's.
  template <typename S>
  static std::vector<int64_t> I64List(const S& input) {
    return torch::lazy::ToVector<int64_t>(input);
  }

  static std::optional<int64_t> I64Optional(std::optional<int64_t> opt) {
    return opt ? std::optional<int64_t>(*opt) : std::nullopt;
  }

  // Creates an XLA padding configuration from a n-dimensional padding list.
  static xla::PaddingConfig MakeXlaPaddingConfigFromNdPadding(
      absl::Span<const int64_t> padding);

  // Retrieves the dynamic dimension of an input shape, or returns -1 if none.
  static int64_t GetDynamicDimension(const xla::Shape& shape);

  static DynamicSize GetDimensionsSize(absl::Span<const xla::XlaOp> inputs,
                                       absl::Span<const int64_t> dimensions);

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

  static xla::XlaComputation CreateMaxAndArgMaxComputation(
      xla::PrimitiveType value_type, xla::PrimitiveType index_type);

  // Returns an XLA operation which is a reshape to the expected rank, by
  // appending 1s to the major dimension. If offset is greater than zero, 1s
  // will be prepened to the minor dimension as well.
  // Expected condition: rank(input) + offset <= expected_rank
  static xla::XlaOp ReshapeToRank(xla::XlaOp input, int64_t expected_rank,
                                  int64_t offset = 0);

  static xla::XlaOp Flatten(xla::XlaOp input,
                            xla::Shape* input_shape = nullptr);

  static xla::XlaOp FlattenDimRange(xla::XlaOp input, int64_t start,
                                    int64_t range,
                                    xla::Shape* input_shape = nullptr);

  // Gathers the input using the order specified by the permutation. For each i,
  // output[i] = input[permutation[i]]. The given permutation must be the same
  // size as the input.
  template <typename Container>
  static std::vector<typename Container::value_type> Permute(
      absl::Span<const int64_t> permutation, const Container& input) {
    using T = typename Container::value_type;
    XLA_CHECK(input.size() == permutation.size() &&
              xla::IsPermutation(permutation))
        << "Invalid permutation specified";
    std::vector<T> output(input.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
      output[i] = input[permutation[i]];
    }
    return output;
  }

  static xla::PrimitiveType PromoteType(xla::PrimitiveType type1,
                                        xla::PrimitiveType type2);

  static xla::PrimitiveType PromoteType(xla::PrimitiveType type1,
                                        xla::PrimitiveType type2,
                                        xla::PrimitiveType type3);

  // Performs type promotion to make sure both operations return the same type.
  static std::pair<xla::XlaOp, xla::XlaOp> PromoteValues(xla::XlaOp op1,
                                                         xla::XlaOp op2);

  static std::tuple<xla::XlaOp, xla::XlaOp, xla::XlaOp> PromoteValues(
      xla::XlaOp op1, xla::XlaOp op2, xla::XlaOp op3);

  // Performs type promotion, by casting the second operation to the type of the
  // first, if different.
  static std::pair<xla::XlaOp, xla::XlaOp> PromoteSecondValue(xla::XlaOp op1,
                                                              xla::XlaOp op2);

  // If any of the shapes of input operations has unbounded dynamic dimensions,
  // performs implicit broadcasting and return the broadcasted operations. For
  // static or bounded dynamic input shapes, validate the shapes and return the
  // input operations. The implicit broadcasting in static and bounded dynamic
  // cases will be handled eventually by the XlaBuilder.
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

  // Given the two shape 'shape1' and 'shape2', infers the broadcasted shape.
  static absl::StatusOr<xla::Shape> GetPromotedShape(const xla::Shape& shape1,
                                                     const xla::Shape& shape2);

  static xla::Shape GetPromotedDynamicShape(const xla::Shape& shape1,
                                            const xla::Shape& shape2);

  // TODO @wonjoo - Migrate to torch::lazy after Shape is migrated
  static xla::Shape GetPromotedBinaryOpShape(const xla::Shape& shape1,
                                             const xla::Shape& shape2);

  // Returns a new operations which broadcast the input operation into the
  // shape. The op_shape is the shape of the op operation, while shape should be
  // one that op is broadcast-able to (usually the result of a
  // GetPromotedShape() call). If op_shape matches shape, the op itself is
  // returned.
  static xla::XlaOp ImplicitBroadcast(xla::XlaOp op, const xla::Shape& op_shape,
                                      const xla::Shape& shape);

  // Returns new operations which broadcast the input operations 'op1' and 'op2'
  // with unbounded dynamic dimensions into the 'shape' which is usually the
  // result of a GetPromotedShape() call.
  // Assumption: The shapes of 'op1' and 'op2' are valid for broadcasting.
  // TODO: We need to emit runtime shape assertions to validate the broadcasting
  // rules are met.
  static std::pair<xla::XlaOp, xla::XlaOp>
  ImplicitBroadcastWithUnboundedDynamicShapes(xla::XlaOp op1, xla::XlaOp op2,
                                              const xla::Shape& shape);

  // Retuns the explicit broadcasting specifications on operations between
  // arrays of different ranks.
  static std::vector<int64_t> getBroadcastDimensions(xla::XlaOp op1,
                                                     xla::XlaOp op2);

  // Performs the bin_op binary operation by promoting types and shapes of the
  // two input operands.
  static xla::XlaOp PromotedBinaryOp(
      xla::XlaOp op1, xla::XlaOp op2,
      const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& bin_op);

  // Basic promoted binary operation implementation follow.
  static xla::XlaOp PromotedAdd(xla::XlaOp op1, xla::XlaOp op2) {
    return PromotedBinaryOp(op1, op2, [](xla::XlaOp op1, xla::XlaOp op2) {
      return xla::Add(op1, op2, getBroadcastDimensions(op1, op2));
    });
  }

  static xla::XlaOp PromotedSub(xla::XlaOp op1, xla::XlaOp op2) {
    return PromotedBinaryOp(op1, op2, [](xla::XlaOp op1, xla::XlaOp op2) {
      return xla::Sub(op1, op2, getBroadcastDimensions(op1, op2));
    });
  }

  static xla::XlaOp PromotedMul(xla::XlaOp op1, xla::XlaOp op2) {
    return PromotedBinaryOp(op1, op2, [](xla::XlaOp op1, xla::XlaOp op2) {
      return xla::Mul(op1, op2, getBroadcastDimensions(op1, op2));
    });
  }

  static xla::XlaOp PromotedDiv(xla::XlaOp op1, xla::XlaOp op2) {
    return PromotedBinaryOp(op1, op2, [](xla::XlaOp op1, xla::XlaOp op2) {
      return xla::Div(op1, op2, getBroadcastDimensions(op1, op2));
    });
  }

  static xla::XlaOp PromotedLogicalBinaryOp(
      xla::XlaOp op1, xla::XlaOp op2,
      const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& bin_op);

  static xla::XlaOp PromotedLogicalUnaryOp(
      xla::XlaOp op, const std::function<xla::XlaOp(xla::XlaOp)>& unary_op);

  // T is the returned type, A is the type used for accumulation. In general,
  // A should have higher-or-equal-precision to T.
  template <typename T, typename A = T>
  static xla::Literal Range(A start, A end, A step) {
    std::vector<A> accumulated = runtime::util::Range<A>(start, end, step);
    return xla::LiteralUtil::CreateR1<T>(
        std::vector<T>(accumulated.begin(), accumulated.end()));
  }

  static xla::PrecisionConfig::Precision mat_mul_precision() {
    return s_mat_mul_precision;
  }

  static void set_mat_mul_precision(xla::PrecisionConfig::Precision precision) {
    s_mat_mul_precision = precision;
  }

  static absl::StatusOr<xla::XlaComputation> WrapXlaComputation(
      const xla::XlaComputation& computation,
      const std::vector<xla::Shape>& parameter_shapes,
      const std::vector<xla::HloSharding>& parameter_shardings,
      const std::vector<size_t>& buffer_donor_indices);

  static std::vector<xla::HloSharding> ExtractInputShardings(
      const xla::XlaComputation& computation);

  static torch::lazy::Shape ConvertXlaShapeToLazy(const xla::Shape& shape);

 private:
  static xla::PrecisionConfig::Precision s_mat_mul_precision;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_HELPERS_H_
