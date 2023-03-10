#include "torch_xla/csrc/reduction.h"

#include <ATen/core/Reduction.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/util.h>

#include <cmath>
#include <unordered_set>

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "third_party/xla_client/debug_macros.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ops/einsum_utilities.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

struct ReductionInfo {
  std::vector<int64_t> new_dimensions;
  XlaHelpers::DynamicSize element_count;
};

struct SummationResult {
  ReductionInfo rinfo;
  xla::XlaOp result;
};

ReductionInfo GetReductionInfo(xla::XlaOp input, const xla::Shape& shape,
                               absl::Span<const int64_t> dimensions,
                               bool keep_reduced_dimensions) {
  ReductionInfo rinfo;
  std::unordered_set<int64_t> reduced_dimensions(dimensions.begin(),
                                                 dimensions.end());
  for (int64_t i = 0; i < shape.rank(); ++i) {
    if (reduced_dimensions.count(i) > 0) {
      if (keep_reduced_dimensions) {
        rinfo.new_dimensions.push_back(1);
      }
    } else {
      rinfo.new_dimensions.push_back(shape.dimensions(i));
    }
  }
  rinfo.element_count = XlaHelpers::GetDimensionsSize({input}, dimensions);
  return rinfo;
}

xla::XlaComputation CreateAllComputation(xla::PrimitiveType type) {
  xla::XlaBuilder builder("AllComputation");
  xla::XlaOp x =
      xla::Parameter(&builder, 0, xla::ShapeUtil::MakeShape(type, {}), "x");
  xla::XlaOp y =
      xla::Parameter(&builder, 1, xla::ShapeUtil::MakeShape(type, {}), "y");
  xla::XlaOp zero = xla::Zero(&builder, type);
  xla::XlaOp one = xla::One(&builder, type);
  xla::Select(xla::And(xla::Ne(x, zero), xla::Ne(y, zero)), one, zero);
  return ConsumeValue(builder.Build());
}

xla::XlaComputation CreateAnyComputation(xla::PrimitiveType type) {
  xla::XlaBuilder builder("AnyComputation");
  xla::XlaOp x =
      xla::Parameter(&builder, 0, xla::ShapeUtil::MakeShape(type, {}), "x");
  xla::XlaOp y =
      xla::Parameter(&builder, 1, xla::ShapeUtil::MakeShape(type, {}), "y");
  xla::XlaOp zero = xla::Zero(&builder, type);
  xla::XlaOp one = xla::One(&builder, type);
  xla::Select(xla::Or(xla::Ne(x, zero), xla::Ne(y, zero)), one, zero);
  return ConsumeValue(builder.Build());
}

xla::XlaOp GetScaleValue(xla::XlaOp input, xla::XlaOp count,
                         xla::PrimitiveType type) {
  xla::XlaOp zero = xla::Zero(input.builder(), XlaHelpers::TypeOfXlaOp(count));
  xla::XlaOp one = xla::One(input.builder(), type);
  xla::XlaOp scale = xla::Select(xla::Ne(count, zero),
                                 one / xla::ConvertElementType(count, type),
                                 xla::NanValue(input.builder(), type));
  return input * scale;
}

xla::XlaOp AverageValue(xla::XlaOp input, xla::XlaOp reduced) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp num_elements =
      XlaHelpers::GetDimensionsSize({input},
                                    XlaHelpers::GetAllDimensions(input_shape))
          .size;
  return GetScaleValue(reduced, num_elements, input_shape.element_type());
}

SummationResult CreateSummation(xla::XlaOp input,
                                absl::Span<const int64_t> dimensions,
                                bool keep_reduced_dimensions, bool scale) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp init_value = xla::Zero(input.builder(), shape.element_type());
  SummationResult result;
  result.rinfo =
      GetReductionInfo(input, shape, dimensions, keep_reduced_dimensions);
  result.result = xla::Reduce(
      input, init_value, XlaHelpers::CreateAddComputation(shape.element_type()),
      dimensions);
  if (scale) {
    result.result = GetScaleValue(
        result.result, result.rinfo.element_count.size, shape.element_type());
  }
  if (keep_reduced_dimensions) {
    result.result =
        XlaHelpers::DynamicReshape(result.result, result.rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp CreateProduct(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                         bool keep_reduced_dimensions) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp init_value = xla::One(input.builder(), shape.element_type());
  ReductionInfo rinfo =
      GetReductionInfo(input, shape, dimensions, keep_reduced_dimensions);
  xla::XlaOp result = xla::Reduce(
      input, init_value, XlaHelpers::CreateMulComputation(shape.element_type()),
      dimensions);
  if (keep_reduced_dimensions) {
    result = XlaHelpers::DynamicReshape(result, rinfo.new_dimensions);
  }
  return result;
}

}  // namespace

ReductionMode GetXlaReductionMode(int64_t reduction) {
  switch (reduction) {
    case at::Reduction::Mean:
      return ReductionMode::kMean;
    case at::Reduction::None:
      return ReductionMode::kNone;
    case at::Reduction::Sum:
      return ReductionMode::kSum;
  }
  XLA_ERROR() << "Unknown reduction mode: " << reduction;
}

xla::XlaOp BuildBinaryCrossEntropy(xla::XlaOp input, xla::XlaOp target,
                                   const absl::optional<xla::XlaOp>& weight,
                                   ReductionMode reduction) {
  static const float kLogBound = -100;
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp xweight;
  if (weight) {
    // PyTorch guards weight and input has the same shape.
    xweight = *weight;
  } else {
    xweight =
        XlaHelpers::ScalarBroadcast<float>(1.0, input_shape, target.builder());
  }
  xla::XlaOp one = xla::One(input.builder(), input_shape.element_type());
  xla::XlaOp log_bound = XlaHelpers::ScalarValue(
      kLogBound, input_shape.element_type(), input.builder());
  xla::XlaOp result =
      -xweight * (target * xla::Max(xla::Log(input), log_bound) +
                  (one - target) * xla::Max(xla::Log(one - input), log_bound));
  if (reduction == ReductionMode::kNone) {
    return result;
  }
  xla::XlaOp reduced_result = xla::ReduceAll(
      result, xla::Zero(input.builder(), input_shape.element_type()),
      XlaHelpers::CreateAddComputation(input_shape.element_type()));
  if (reduction == ReductionMode::kMean) {
    reduced_result = AverageValue(result, reduced_result);
  }
  return reduced_result;
}

xla::XlaOp BuildBinaryCrossEntropyBackward(
    xla::XlaOp grad_output, xla::XlaOp input, xla::XlaOp target,
    const absl::optional<xla::XlaOp>& weight, ReductionMode reduction) {
  static const float kEpsilon = 1e-12;
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp xweight;
  if (weight) {
    // PyTorch guards weight and input has the same shape.
    xweight = *weight;
  } else {
    xweight =
        XlaHelpers::ScalarBroadcast<float>(1.0, input_shape, target.builder());
  }
  xla::XlaOp one = xla::One(input.builder(), input_shape.element_type());
  xla::XlaOp epsilon = XlaHelpers::ScalarValue(
      kEpsilon, input_shape.element_type(), input.builder());
  xla::XlaOp result =
      xweight * (input - target) / xla::Max(input * (one - input), epsilon);
  if (reduction == ReductionMode::kNone) {
    return result * grad_output;
  }
  result = result * grad_output;
  if (reduction == ReductionMode::kMean) {
    result = AverageValue(input, result);
  }
  return result;
}

xla::XlaOp BuildMseLoss(xla::XlaOp input, xla::XlaOp target,
                        ReductionMode reduction) {
  xla::XlaOp diff = XlaHelpers::PromotedSub(input, target);
  xla::XlaOp result = diff * diff;
  if (reduction == ReductionMode::kNone) {
    return result;
  }
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  const xla::Shape& result_shape = XlaHelpers::ShapeOfXlaOp(result);
  result = xla::ReduceAll(
      result, xla::Zero(result.builder(), result_shape.element_type()),
      XlaHelpers::CreateAddComputation(result_shape.element_type()));
  if (reduction == ReductionMode::kMean) {
    int64_t num_elements = xla::ShapeUtil::ElementsIn(input_shape);
    if (num_elements == 0) {
      return xla::NanValue(input.builder(), input_shape.element_type());
    } else {
      xla::XlaOp scale_value = XlaHelpers::ScalarValue<double>(
          1.0 / static_cast<double>(num_elements), result_shape.element_type(),
          result.builder());
      result = result * scale_value;
    }
  }
  return result;
}

xla::XlaOp BuildMseLossBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                xla::XlaOp target, ReductionMode reduction) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp two = XlaHelpers::ScalarValue<double>(
      2, input_shape.element_type(), input.builder());
  xla::XlaOp d_input =
      XlaHelpers::PromotedMul(two, XlaHelpers::PromotedSub(input, target));
  if (reduction == ReductionMode::kNone) {
    return XlaHelpers::PromotedMul(d_input, grad_output);
  }
  xla::XlaOp grad_value = grad_output;
  if (reduction == ReductionMode::kMean) {
    int64_t num_elements = xla::ShapeUtil::ElementsIn(input_shape);
    xla::XlaOp scale_value = XlaHelpers::ScalarValue<double>(
        1.0 / static_cast<double>(num_elements), input_shape.element_type(),
        input.builder());
    grad_value = XlaHelpers::PromotedMul(grad_output, scale_value);
  }
  return XlaHelpers::PromotedMul(d_input, grad_value);
}

xla::XlaOp BuildCumulativeComputation(xla::XlaOp input, int64_t dim,
                                      const xla::XlaComputation& reducer,
                                      xla::XlaOp init) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  std::vector<int64_t> window_strides(input_shape.rank(), 1);
  std::vector<int64_t> window_dims(input_shape.rank(), 1);
  window_dims[dim] = input_shape.dimensions(dim);
  std::vector<std::pair<int64_t, int64_t>> padding(input_shape.rank());
  padding[dim].first = input_shape.dimensions(dim) - 1;
  return xla::ReduceWindowWithGeneralPadding(
      input, init, reducer, window_dims, window_strides,
      /*base_dilations=*/{}, /*window_dilations=*/{}, padding);
}

xla::XlaOp BuildMean(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                     bool keep_reduced_dimensions) {
  return CreateSummation(input, dimensions, keep_reduced_dimensions,
                         /*scale=*/true)
      .result;
}

xla::XlaOp ApplyCorrectedScaling(const SummationResult& sum_result,
                                 double correction, xla::PrimitiveType type) {
  auto builder = sum_result.result.builder();
  auto count_real =
      xla::ConvertElementType(sum_result.rinfo.element_count.size, type);
  auto correction_scalar = XlaHelpers::ScalarValue(correction, type, builder);
  auto zero = xla::Zero(builder, type);
  auto dof = xla::Max(zero, count_real - correction_scalar);
  auto one = xla::One(builder, type);
  auto scale = one / dof;
  return sum_result.result * scale;
}

xla::XlaOp BuildVar(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                    double correction, bool keep_reduced_dimensions) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp mean =
      BuildMean(input, dimensions, /*keep_reduced_dimensions*/ true);
  xla::XlaOp bcast_mean =
      xla::BroadcastInDim(mean, input_shape.dimensions(),
                          torch::lazy::Iota<int64_t>(input_shape.rank()));
  xla::XlaOp input_mean_diff = input - bcast_mean;
  xla::XlaOp diff2 = input_mean_diff * input_mean_diff;
  xla::XlaOp var;
  if (correction != 0) {
    SummationResult sum_result = CreateSummation(
        diff2, dimensions, keep_reduced_dimensions, /*scale=*/false);
    var = ApplyCorrectedScaling(sum_result, correction,
                                input_shape.element_type());
  } else {
    SummationResult sum_result = CreateSummation(
        diff2, dimensions, keep_reduced_dimensions, /*scale=*/true);
    var = sum_result.result;
  }
  return var;
}

xla::XlaOp BuildStdDeviation(xla::XlaOp input,
                             absl::Span<const int64_t> dimensions,
                             bool keep_reduced_dimensions, double correction) {
  auto var = BuildVar(input, dimensions, correction, keep_reduced_dimensions);
  return xla::Sqrt(var);
}

xla::XlaOp BuildSum(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                    bool keep_reduced_dimensions) {
  return CreateSummation(input, dimensions, keep_reduced_dimensions,
                         /*scale=*/false)
      .result;
}

xla::XlaOp BuildProd(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                     bool keep_reduced_dimensions) {
  return CreateProduct(input, dimensions, keep_reduced_dimensions);
}

xla::XlaOp BuildMaxInDim(xla::XlaOp input, int64_t dim,
                         bool keep_reduced_dimensions) {
  return BuildMaxInDims(input, {dim}, keep_reduced_dimensions);
}

xla::XlaOp BuildMaxInDims(xla::XlaOp input,
                          absl::Span<const int64_t> dimensions,
                          bool keep_reduced_dimensions) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  XlaHelpers::MinMax min_max = XlaHelpers::MinMaxValues(shape.element_type());
  std::vector<int64_t> canonical_dimensions =
      torch::lazy::GetCanonicalDimensionIndices(
          xla::util::ToVector<int64_t>(dimensions), shape.rank());
  xla::XlaOp init_value = XlaHelpers::ScalarValue(
      min_max.min, shape.element_type(), input.builder());
  ReductionInfo rinfo = GetReductionInfo(input, shape, canonical_dimensions,
                                         keep_reduced_dimensions);
  if (rinfo.element_count.scalar_size) {
    // When can only assert this if dimensions are not dynamic.
    XLA_CHECK_GT(*rinfo.element_count.scalar_size, 0);
  }
  xla::XlaOp result = xla::Reduce(
      input, init_value, XlaHelpers::CreateMaxComputation(shape.element_type()),
      canonical_dimensions);
  if (keep_reduced_dimensions) {
    result = XlaHelpers::DynamicReshape(result, rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp BuildMinInDim(xla::XlaOp input, int64_t dim,
                         bool keep_reduced_dimensions) {
  return BuildMinInDims(input, {dim}, keep_reduced_dimensions);
}

xla::XlaOp BuildMinInDims(xla::XlaOp input,
                          absl::Span<const int64_t> dimensions,
                          bool keep_reduced_dimensions) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  XlaHelpers::MinMax min_max = XlaHelpers::MinMaxValues(shape.element_type());

  std::vector<int64_t> canonical_dimensions =
      torch::lazy::GetCanonicalDimensionIndices(
          xla::util::ToVector<int64_t>(dimensions), shape.rank());

  xla::XlaOp init_value = XlaHelpers::ScalarValue(
      min_max.max, shape.element_type(), input.builder());
  ReductionInfo rinfo = GetReductionInfo(input, shape, canonical_dimensions,
                                         keep_reduced_dimensions);
  if (rinfo.element_count.scalar_size) {
    // When can only assert this if dimensions are not dynamic.
    XLA_CHECK_GT(*rinfo.element_count.scalar_size, 0);
  }
  xla::XlaOp result = xla::Reduce(
      input, init_value, XlaHelpers::CreateMinComputation(shape.element_type()),
      canonical_dimensions);
  if (keep_reduced_dimensions) {
    result = XlaHelpers::DynamicReshape(result, rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp BuildArgMax(xla::XlaOp input, int64_t dim, bool keepdim) {
  const xla::Shape* shape = &XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp operand = input;
  if (dim < 0) {
    dim = 0;
    operand = XlaHelpers::DynamicReshape(operand,
                                         {xla::ShapeUtil::ElementsIn(*shape)});
    shape = &XlaHelpers::ShapeOfXlaOp(operand);
  }
  xla::XlaOp result = xla::ArgMax(
      operand,
      GetDevicePrimitiveType(xla::PrimitiveType::S64, /*device=*/nullptr), dim);
  if (keepdim) {
    auto dimensions = torch::lazy::ToVector<int64_t>(shape->dimensions());
    dimensions[dim] = 1;
    result = XlaHelpers::DynamicReshape(result, dimensions);
  }
  return result;
}

xla::XlaOp BuildArgMin(xla::XlaOp input, int64_t dim, bool keepdim) {
  const xla::Shape* shape = &XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp operand = input;
  if (dim < 0) {
    dim = 0;
    operand = XlaHelpers::DynamicReshape(operand,
                                         {xla::ShapeUtil::ElementsIn(*shape)});
    shape = &XlaHelpers::ShapeOfXlaOp(operand);
  }
  xla::XlaOp result = xla::ArgMin(
      operand,
      GetDevicePrimitiveType(xla::PrimitiveType::S64, /*device=*/nullptr), dim);
  if (keepdim) {
    auto dimensions = torch::lazy::ToVector<int64_t>(shape->dimensions());
    dimensions[dim] = 1;
    result = XlaHelpers::DynamicReshape(result, dimensions);
  }
  return result;
}

xla::XlaOp BuildAll(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                    bool keep_reduced_dimensions) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  std::vector<int64_t> canonical_dimensions =
      torch::lazy::GetCanonicalDimensionIndices(
          xla::util::ToVector<int64_t>(dimensions), shape.rank());
  ReductionInfo rinfo = GetReductionInfo(input, shape, canonical_dimensions,
                                         keep_reduced_dimensions);
  xla::XlaOp init_value = xla::ConstantLiteral(
      input.builder(), xla::LiteralUtil::One(shape.element_type()));
  xla::PrimitiveType result_type =
      shape.element_type() == xla::PrimitiveType::U8 ? xla::PrimitiveType::U8
                                                     : xla::PrimitiveType::PRED;
  xla::XlaOp result =
      xla::Reduce(input, init_value, CreateAllComputation(shape.element_type()),
                  canonical_dimensions);
  result = MaybeConvertTo(
      xla::Ne(result, xla::Zero(input.builder(), shape.element_type())),
      result_type);
  if (keep_reduced_dimensions) {
    result = XlaHelpers::DynamicReshape(result, rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp BuildAny(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                    bool keep_reduced_dimensions) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  std::vector<int64_t> canonical_dimensions =
      torch::lazy::GetCanonicalDimensionIndices(
          xla::util::ToVector<int64_t>(dimensions), shape.rank());
  ReductionInfo rinfo = GetReductionInfo(input, shape, canonical_dimensions,
                                         keep_reduced_dimensions);
  xla::XlaOp init_value = xla::ConstantLiteral(
      input.builder(), xla::LiteralUtil::Zero(shape.element_type()));
  xla::PrimitiveType result_type =
      shape.element_type() == xla::PrimitiveType::U8 ? xla::PrimitiveType::U8
                                                     : xla::PrimitiveType::PRED;
  xla::XlaOp result =
      xla::Reduce(input, init_value, CreateAnyComputation(shape.element_type()),
                  canonical_dimensions);
  result = MaybeConvertTo(
      xla::Ne(result, xla::Zero(input.builder(), shape.element_type())),
      result_type);
  if (keep_reduced_dimensions) {
    result = XlaHelpers::DynamicReshape(result, rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp BuildLogsumexp(xla::XlaOp input,
                          absl::Span<const int64_t> dimensions,
                          bool keep_reduced_dimensions) {
  // Use the log-sum-exp trick to avoid overflow.
  xla::XlaOp max_in_dim =
      BuildMaxInDims(input, dimensions, /*keep_reduced_dimensions=*/true);
  xla::XlaOp exps = xla::Exp(input - max_in_dim);
  xla::XlaOp sums = CreateSummation(exps, dimensions, keep_reduced_dimensions,
                                    /*scale=*/false)
                        .result;
  xla::XlaOp logs = xla::Log(sums);
  // If keep_reduced_dimensions is false, we need to reshape the max_in_dim to
  // the reduced shape before doing the add.
  if (!keep_reduced_dimensions) {
    max_in_dim =
        CreateSummation(max_in_dim, dimensions, keep_reduced_dimensions,
                        /*scale=*/false)
            .result;
  }
  return logs + max_in_dim;
}

xla::XlaOp BuildEinsum(absl::Span<const xla::XlaOp> operands,
                       const std::string& equation) {
  if (operands.size() == 1) {
    return xla::Einsum(
        operands[0], equation,
        xla::PrecisionConfig::Precision::PrecisionConfig_Precision_DEFAULT);
  } else if (operands.size() == 2) {
    return xla::Einsum(
        operands[0], operands[1], equation,
        xla::PrecisionConfig::Precision::PrecisionConfig_Precision_DEFAULT,
        XlaHelpers::PromoteType(XlaHelpers::TypeOfXlaOp(operands[0]),
                                XlaHelpers::TypeOfXlaOp(operands[1])));
  }
}

std::vector<xla::XlaOp> BuildEinsumBackward(const xla::XlaOp& grad_output,
                                            absl::Span<const xla::XlaOp> inputs,
                                            const std::string& equation) {
  std::vector<xla::XlaOp> result;
  if (inputs.size() == 1) {
    std::string backward_equation =
        EinsumUtilities::BuildBackwardsEquation(equation);
    result.push_back(xla::Einsum(grad_output, backward_equation));
  } else if (inputs.size() == 2) {
    std::vector<std::string> equations =
        EinsumUtilities::BuildBackwardsEquations(equation);

    xla::PrimitiveType type = XlaHelpers::PromoteType(
        XlaHelpers::TypeOfXlaOp(grad_output),
        XlaHelpers::TypeOfXlaOp(inputs[0]), XlaHelpers::TypeOfXlaOp(inputs[1]));

    result.push_back(xla::Einsum(
        grad_output, inputs[1], equations[0],
        xla::PrecisionConfig::Precision::PrecisionConfig_Precision_DEFAULT,
        type));

    result.push_back(xla::Einsum(
        inputs[0], grad_output, equations[1],
        xla::PrecisionConfig::Precision::PrecisionConfig_Precision_DEFAULT,
        type));
  }

  return result;
}

}  // namespace torch_xla
