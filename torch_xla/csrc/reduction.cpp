#include "torch_xla/csrc/reduction.h"

#include <cmath>
#include <unordered_set>

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

struct ReductionInfo {
  std::vector<xla::int64> new_dimensions;
  XlaHelpers::DynamicSize element_count;
};

struct SummationResult {
  ReductionInfo rinfo;
  xla::XlaOp result;
};

ReductionInfo GetReductionInfo(xla::XlaOp input, const xla::Shape& shape,
                               absl::Span<const xla::int64> dimensions,
                               bool keep_reduced_dimensions) {
  ReductionInfo rinfo;
  std::unordered_set<xla::int64> reduced_dimensions(dimensions.begin(),
                                                    dimensions.end());
  for (xla::int64 i = 0; i < shape.rank(); ++i) {
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
                                absl::Span<const xla::int64> dimensions,
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

xla::XlaOp CreateProduct(xla::XlaOp input,
                         absl::Span<const xla::int64> dimensions,
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

xla::XlaOp BuildL1Loss(xla::XlaOp input, xla::XlaOp target,
                       ReductionMode reduction) {
  xla::XlaOp result = xla::Abs(input - target);
  if (reduction == ReductionMode::kNone) {
    return result;
  }
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  result = xla::ReduceAll(
      result, xla::Zero(input.builder(), input_shape.element_type()),
      XlaHelpers::CreateAddComputation(input_shape.element_type()));
  if (reduction == ReductionMode::kMean) {
    result = AverageValue(input, result);
  }
  return result;
}

xla::XlaOp BuildL1LossBackward(xla::XlaOp grad_output, xla::XlaOp input,
                               xla::XlaOp target, ReductionMode reduction) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  if (reduction == ReductionMode::kNone) {
    xla::XlaOp one = xla::One(input.builder(), input_shape.element_type());
    xla::XlaOp mask = xla::Select(xla::Ge(input, target), one, -one);
    return mask * grad_output;
  }
  xla::XlaOp grad_value = grad_output;
  if (reduction == ReductionMode::kMean) {
    grad_value = AverageValue(input, grad_value);
  }
  return xla::Select(xla::Ge(input, target), grad_value, -grad_value);
}

xla::XlaOp BuildMseLoss(xla::XlaOp input, xla::XlaOp target,
                        ReductionMode reduction) {
  xla::XlaOp diff = input - target;
  xla::XlaOp result = diff * diff;
  if (reduction == ReductionMode::kNone) {
    return result;
  }
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  result = xla::ReduceAll(
      result, xla::Zero(input.builder(), input_shape.element_type()),
      XlaHelpers::CreateAddComputation(input_shape.element_type()));
  if (reduction == ReductionMode::kMean) {
    xla::int64 num_elements = xla::ShapeUtil::ElementsIn(input_shape);
    if (num_elements == 0) {
      return xla::NanValue(input.builder(), input_shape.element_type());
    } else {
      xla::XlaOp scale_value = XlaHelpers::ScalarValue<double>(
          1.0 / static_cast<double>(num_elements), input_shape.element_type(),
          input.builder());
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
  xla::XlaOp d_input = two * (input - target);
  if (reduction == ReductionMode::kNone) {
    return d_input * grad_output;
  }
  xla::XlaOp grad_value = grad_output;
  if (reduction == ReductionMode::kMean) {
    xla::int64 num_elements = xla::ShapeUtil::ElementsIn(input_shape);
    xla::XlaOp scale_value = XlaHelpers::ScalarValue<double>(
        1.0 / static_cast<double>(num_elements), input_shape.element_type(),
        input.builder());
    grad_value = grad_output * scale_value;
  }
  return d_input * grad_value;
}

xla::XlaOp BuildCumulativeComputation(xla::XlaOp input, xla::int64 dim,
                                      const xla::XlaComputation& reducer,
                                      xla::XlaOp init) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  std::vector<xla::int64> window_strides(input_shape.rank(), 1);
  std::vector<xla::int64> window_dims(input_shape.rank(), 1);
  window_dims[dim] = input_shape.dimensions(dim);
  std::vector<std::pair<xla::int64, xla::int64>> padding(input_shape.rank());
  padding[dim].first = input_shape.dimensions(dim) - 1;
  return xla::ReduceWindowWithGeneralPadding(
      input, init, reducer, window_dims, window_strides,
      /*base_dilations=*/{}, /*window_dilations=*/{}, padding);
}

xla::XlaOp BuildMean(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                     bool keep_reduced_dimensions) {
  return CreateSummation(input, dimensions, keep_reduced_dimensions,
                         /*scale=*/true)
      .result;
}

xla::XlaOp BuildStdDeviation(xla::XlaOp input,
                             absl::Span<const xla::int64> dimensions,
                             bool keep_reduced_dimensions, bool unbiased) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp mean =
      BuildMean(input, dimensions, /*keep_reduced_dimensions*/ true);
  xla::XlaOp bcast_mean =
      xla::BroadcastInDim(mean, input_shape.dimensions(),
                          xla::util::Iota<xla::int64>(input_shape.rank()));
  xla::XlaOp input_mean_diff = input - bcast_mean;
  xla::XlaOp squared_var = input_mean_diff * input_mean_diff;
  xla::XlaOp squared_result;
  if (unbiased) {
    SummationResult sum_result = CreateSummation(
        squared_var, dimensions, keep_reduced_dimensions, /*scale=*/false);
    squared_result = GetScaleValue(
        sum_result.result,
        sum_result.rinfo.element_count.size -
            xla::One(input.builder(), XlaHelpers::TypeOfXlaOp(
                                          sum_result.rinfo.element_count.size)),
        input_shape.element_type());
  } else {
    SummationResult sum_result = CreateSummation(
        squared_var, dimensions, keep_reduced_dimensions, /*scale=*/true);
    squared_result = sum_result.result;
  }
  return xla::Sqrt(squared_result);
}

xla::XlaOp BuildSum(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                    bool keep_reduced_dimensions) {
  return CreateSummation(input, dimensions, keep_reduced_dimensions,
                         /*scale=*/false)
      .result;
}

xla::XlaOp BuildProd(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                     bool keep_reduced_dimensions) {
  return CreateProduct(input, dimensions, keep_reduced_dimensions);
}

xla::XlaOp BuildMaxInDim(xla::XlaOp input, xla::int64 dim,
                         bool keep_reduced_dimensions) {
  return BuildMaxInDims(input, {dim}, keep_reduced_dimensions);
}

xla::XlaOp BuildMaxInDims(xla::XlaOp input,
                          absl::Span<const xla::int64> dimensions,
                          bool keep_reduced_dimensions) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  XlaHelpers::MinMax min_max = XlaHelpers::MinMaxValues(shape.element_type());
  xla::XlaOp init_value = XlaHelpers::ScalarValue(
      min_max.min, shape.element_type(), input.builder());
  ReductionInfo rinfo =
      GetReductionInfo(input, shape, dimensions, keep_reduced_dimensions);
  if (rinfo.element_count.scalar_size) {
    // When can only assert this if dimensions are not dynamic.
    XLA_CHECK_GT(*rinfo.element_count.scalar_size, 0);
  }
  xla::XlaOp result = xla::Reduce(
      input, init_value, XlaHelpers::CreateMaxComputation(shape.element_type()),
      dimensions);
  if (keep_reduced_dimensions) {
    result = XlaHelpers::DynamicReshape(result, rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp BuildMinInDim(xla::XlaOp input, xla::int64 dim,
                         bool keep_reduced_dimensions) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  XlaHelpers::MinMax min_max = XlaHelpers::MinMaxValues(shape.element_type());
  xla::XlaOp init_value = XlaHelpers::ScalarValue(
      min_max.max, shape.element_type(), input.builder());
  ReductionInfo rinfo =
      GetReductionInfo(input, shape, {dim}, keep_reduced_dimensions);
  if (rinfo.element_count.scalar_size) {
    // When can only assert this if dimensions are not dynamic.
    XLA_CHECK_GT(*rinfo.element_count.scalar_size, 0);
  }
  xla::XlaOp result = xla::Reduce(
      input, init_value, XlaHelpers::CreateMinComputation(shape.element_type()),
      {dim});
  if (keep_reduced_dimensions) {
    result = XlaHelpers::DynamicReshape(result, rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp BuildArgMax(xla::XlaOp input, xla::int64 dim, bool keepdim) {
  const xla::Shape* shape = &XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp operand = input;
  if (dim < 0) {
    dim = 0;
    operand = XlaHelpers::DynamicReshape(operand,
                                         {xla::ShapeUtil::ElementsIn(*shape)});
    shape = &XlaHelpers::ShapeOfXlaOp(operand);
  }
  xla::XlaOp result = xla::ArgMaxTwoPass(
      operand,
      GetDevicePrimitiveType(xla::PrimitiveType::S64, /*device=*/nullptr), dim,
      /*tie_low=*/true);
  if (keepdim) {
    auto dimensions = xla::util::ToVector<xla::int64>(shape->dimensions());
    dimensions[dim] = 1;
    result = XlaHelpers::DynamicReshape(result, dimensions);
  }
  return result;
}

xla::XlaOp BuildArgMin(xla::XlaOp input, xla::int64 dim, bool keepdim) {
  const xla::Shape* shape = &XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp operand = input;
  if (dim < 0) {
    dim = 0;
    operand = XlaHelpers::DynamicReshape(operand,
                                         {xla::ShapeUtil::ElementsIn(*shape)});
    shape = &XlaHelpers::ShapeOfXlaOp(operand);
  }
  xla::XlaOp result = xla::ArgMinTwoPass(
      operand,
      GetDevicePrimitiveType(xla::PrimitiveType::S64, /*device=*/nullptr), dim,
      /*tie_low=*/true);
  if (keepdim) {
    auto dimensions = xla::util::ToVector<xla::int64>(shape->dimensions());
    dimensions[dim] = 1;
    result = XlaHelpers::DynamicReshape(result, dimensions);
  }
  return result;
}

xla::XlaOp BuildAll(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                    bool keep_reduced_dimensions) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  ReductionInfo rinfo =
      GetReductionInfo(input, shape, dimensions, keep_reduced_dimensions);
  xla::XlaOp init_value = xla::ConstantLiteral(
      input.builder(), xla::LiteralUtil::One(shape.element_type()));
  xla::XlaOp result =
      xla::Reduce(input, init_value, CreateAllComputation(shape.element_type()),
                  dimensions);
  result = xla::Ne(result, xla::Zero(input.builder(), shape.element_type()));
  if (keep_reduced_dimensions) {
    result = XlaHelpers::DynamicReshape(result, rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp BuildAny(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                    bool keep_reduced_dimensions) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  ReductionInfo rinfo =
      GetReductionInfo(input, shape, dimensions, keep_reduced_dimensions);
  xla::XlaOp init_value = xla::ConstantLiteral(
      input.builder(), xla::LiteralUtil::Zero(shape.element_type()));
  xla::XlaOp result =
      xla::Reduce(input, init_value, CreateAnyComputation(shape.element_type()),
                  dimensions);
  result = xla::Ne(result, xla::Zero(input.builder(), shape.element_type()));
  if (keep_reduced_dimensions) {
    result = XlaHelpers::DynamicReshape(result, rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp BuildVar(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                    bool unbiased, bool keep_reduced_dimensions) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  SummationResult mean_result =
      CreateSummation(input, dimensions, /*keep_reduced_dimensions=*/true,
                      /*scale=*/true);
  // var = ((input - mean)^2).sum(dim) / reduced_element_count
  xla::XlaOp diff = input - mean_result.result;
  xla::XlaOp unscaled_result =
      CreateSummation(diff * diff, dimensions, keep_reduced_dimensions,
                      /*scale=*/false)
          .result;
  xla::XlaOp count = mean_result.rinfo.element_count.size;
  if (unbiased) {
    count = count - xla::One(input.builder(),
                             XlaHelpers::ShapeOfXlaOp(count).element_type());
  }
  return GetScaleValue(unscaled_result, count, input_shape.element_type());
}

xla::XlaOp BuildLogsumexp(xla::XlaOp input,
                          absl::Span<const xla::int64> dimensions,
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

}  // namespace torch_xla
