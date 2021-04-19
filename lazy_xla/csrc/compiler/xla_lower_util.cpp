#include "lazy_xla/csrc/compiler/xla_lower_util.h"

#include "lazy_xla/csrc/compiler/data_ops.h"
#include "lazy_xla/csrc/compiler/helpers.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"

namespace torch_lazy_tensors {
namespace compiler {
namespace {

bool ShouldUseDenseScatter(const Device& device, const xla::Shape& input_shape,
                           const xla::Shape& index_shape) {
  static int dense_scatter_factor =
      xla::sys_util::GetEnvInt("XLA_DENSE_SCATTER_FACTOR", 100);
  if (device.hw_type == DeviceType::TPU) {
    xla::int64 input_elements = xla::ShapeUtil::ElementsIn(input_shape);
    xla::int64 index_elements = xla::ShapeUtil::ElementsIn(index_shape);
    return index_elements * dense_scatter_factor >= input_elements;
  }
  return false;
}

xla::XlaComputation MakeScatterComputation(
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& combiner,
    xla::PrimitiveType element_type) {
  xla::XlaBuilder cb("ScatterCombiner");
  xla::Shape xla_scalar_shape = xla::ShapeUtil::MakeShape(element_type, {});
  xla::XlaOp p0 = xla::Parameter(&cb, 0, xla_scalar_shape, "p0");
  xla::XlaOp result = xla::Parameter(&cb, 1, xla_scalar_shape, "p1");
  if (combiner != nullptr) {
    result = combiner(p0, result);
  }
  return ConsumeValue(cb.Build(result));
}

bool ScatterRequiresPadding(const xla::Shape& input_shape,
                            const xla::Shape& index_shape, xla::int64 dim) {
  bool requires_padding = false;
  for (size_t i = 0; i < input_shape.rank(); ++i) {
    if (input_shape.dimensions(i) > index_shape.dimensions(i)) {
      requires_padding = true;
    } else if (i != dim) {
      XLA_CHECK_EQ(input_shape.dimensions(i), index_shape.dimensions(i));
    }
  }
  return requires_padding;
}

xla::XlaOp XlaDenseScatter(xla::XlaOp input, xla::XlaOp index, xla::XlaOp src,
                           xla::int64 dim, const ScatterOptions& options) {
  // Contribute back this code to xla::TorchScatterDense() once this has reached
  // a stable implementation.
  xla::XlaBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
    const xla::Shape& index_shape = XlaHelpers::ShapeOfXlaOp(index);
    const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
    std::vector<xla::int64> index_broacast_dims;
    std::vector<xla::int64> sizes;
    for (xla::int64 i = 0; i < index_shape.rank(); ++i) {
      if (i < dim) {
        index_broacast_dims.push_back(i);
      } else {
        if (i == dim) {
          sizes.push_back(input_shape.dimensions(i));
        }
        index_broacast_dims.push_back(i + 1);
      }
      sizes.push_back(index_shape.dimensions(i));
    }

    xla::XlaOp init_value =
        options.init_value
            ? *options.init_value
            : xla::Zero(input.builder(), input_shape.element_type());
    xla::XlaComputation reduce_computation =
        options.combiner != nullptr
            ? MakeScatterComputation(options.combiner,
                                     input_shape.element_type())
            : xla::CreateScalarIdentityWithZeroComputation(
                  input_shape.element_type(), builder);
    xla::XlaOp mask = xla::Eq(
        xla::BroadcastInDim(index, sizes, index_broacast_dims),
        xla::Iota(builder,
                  xla::ShapeUtil::MakeShape(index_shape.element_type(), sizes),
                  dim));
    xla::XlaOp selected_src =
        xla::Select(mask, xla::BroadcastInDim(src, sizes, index_broacast_dims),
                    xla::Broadcast(init_value, sizes));
    xla::XlaOp masked_src =
        xla::Reduce(selected_src, init_value, reduce_computation, {dim + 1});
    if (options.indices_are_unique &&
        XlaHelpers::SameStaticDimensions(index_shape, input_shape)) {
      // If the index shape is the same as the input shape, the input shape will
      // be fully covered (since scatter indices must be unique), so there is no
      // need for masking.
      return options.combiner != nullptr ? options.combiner(input, masked_src)
                                         : masked_src;
    }
    xla::XlaOp reduced_mask = xla::Reduce(
        mask, xla::ConstantR0<bool>(builder, false),
        xla::CreateScalarOrComputation(xla::PrimitiveType::PRED, builder),
        {dim + 1});
    if (ScatterRequiresPadding(input_shape, index_shape, dim)) {
      masked_src = PadToSize(masked_src, input_shape.dimensions(), init_value);
      reduced_mask = PadToSize(reduced_mask, input_shape.dimensions());
    }
    xla::XlaOp result;
    if (options.combiner != nullptr) {
      result =
          xla::Select(reduced_mask, options.combiner(input, masked_src), input);
    } else {
      result = xla::Select(reduced_mask, masked_src, input);
    }
    return result;
  });
}

}  // namespace

xla::XlaOp PadToSize(xla::XlaOp input, absl::Span<const xla::int64> size,
                     absl::optional<xla::XlaOp> pad_value) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  XLA_CHECK_EQ(input_shape.rank(), size.size());
  if (!pad_value) {
    pad_value = xla::Zero(input.builder(), input_shape.element_type());
  }
  bool has_padding = false;
  xla::PaddingConfig padding_config;
  for (size_t i = 0; i < size.size(); i++) {
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(0);
    dims->set_interior_padding(0);
    dims->set_edge_padding_high(size[i] - input_shape.dimensions(i));
    has_padding = has_padding || dims->edge_padding_high() != 0;
  }
  return has_padding ? xla::Pad(input, *pad_value, padding_config) : input;
}

xla::XlaOp BuildMatMul(xla::XlaOp lhs, xla::XlaOp rhs, xla::XlaOp bias) {
  xla::XlaOp dot = BuildDot(lhs, rhs);
  const xla::Shape& dot_shape = XlaHelpers::ShapeOfXlaOp(dot);
  const xla::Shape& bias_shape = XlaHelpers::ShapeOfXlaOp(bias);
  if (bias_shape.dimensions() != dot_shape.dimensions()) {
    bias = BuildExpand(bias, dot_shape.dimensions());
  }
  return dot + bias;
}

xla::XlaOp BuildDot(xla::XlaOp lhs, xla::XlaOp rhs) {
  xla::PrecisionConfig precision_config =
      XlaHelpers::BuildPrecisionConfig(XlaHelpers::mat_mul_precision());
  return xla::Dot(lhs, rhs, &precision_config);
}

xla::XlaOp CreateScatter(const Device& device, xla::XlaOp input,
                         xla::XlaOp index, xla::XlaOp source, xla::int64 dim,
                         const ScatterOptions& options) {
  const xla::Shape& input_shape = compiler::XlaHelpers::ShapeOfXlaOp(input);
  xla::Shape index_shape = compiler::XlaHelpers::ShapeOfXlaOp(index);
  const xla::Shape& source_shape = compiler::XlaHelpers::ShapeOfXlaOp(source);
  LTC_CHECK_EQ(source_shape.rank(), index_shape.rank());
  xla::XlaOp source_op = source;
  if (source_shape.dimensions() != index_shape.dimensions()) {
    std::vector<xla::int64> base_indices(source_shape.rank(), 0);
    source_op = BuildSlice(source_op, base_indices, index_shape.dimensions());
  }
  if (ShouldUseDenseScatter(device, input_shape, index_shape)) {
    return XlaDenseScatter(input, index, source_op, dim, options);
  }

  xla::ShapeUtil::AppendMajorDimension(1, &index_shape);
  std::vector<xla::XlaOp> to_concat;
  to_concat.reserve(input_shape.rank());
  for (xla::int64 i = 0; i < input_shape.rank(); ++i) {
    if (i == dim) {
      to_concat.push_back(compiler::XlaHelpers::DynamicReshape(
          index, index_shape.dimensions()));
    } else {
      to_concat.push_back(xla::Iota(input.builder(), index_shape, i));
    }
  }
  xla::XlaOp scatter_indices =
      xla::ConcatInDim(input.builder(), to_concat, input_shape.rank());
  xla::ScatterDimensionNumbers scatter_dnums;
  scatter_dnums.set_index_vector_dim(input_shape.rank());
  for (xla::int64 i = 0; i < input_shape.rank(); ++i) {
    scatter_dnums.add_inserted_window_dims(i);
    scatter_dnums.add_scatter_dims_to_operand_dims(i);
  }
  return xla::Scatter(
      input, scatter_indices, source_op,
      MakeScatterComputation(options.combiner, input_shape.element_type()),
      scatter_dnums);
}

XlaOpVector BuildAmpForeachNonFiniteCheckAndUnscale(
    const XlaOpVector& inputs, const xla::XlaOp& found_inf_float,
    const xla::XlaOp& inv_scale) {
  const xla::PrimitiveType origin_type =
      XlaHelpers::ShapeOfXlaOp(found_inf_float).element_type();
  xla::XlaOp one = xla::One(inputs[0].builder(), xla::PrimitiveType::PRED);
  xla::XlaOp found_inf =
      xla::ConvertElementType(found_inf_float, xla::PrimitiveType::PRED);
  for (size_t i = 0; i < inputs.size(); ++i) {
    xla::XlaOp all_finite =
        xla::ReduceAll(xla::IsFinite(inputs[i]), one,
                       xla::CreateScalarAndComputation(xla::PrimitiveType::PRED,
                                                       inputs[i].builder()));
    found_inf = xla::Or(found_inf, xla::Not(all_finite));
  }
  XlaOpVector results;
  for (size_t i = 0; i < inputs.size(); ++i) {
    results.push_back(inputs[i] * inv_scale);
  }
  results.push_back(xla::ConvertElementType(found_inf, origin_type));
  return results;
}

XlaOpVector BuildAmpUpdateScale(const xla::XlaOp& growth_tracker,
                                const xla::XlaOp& current_scale,
                                const xla::XlaOp& found_inf_float,
                                double scale_growth_factor,
                                double scale_backoff_factor,
                                int scale_growth_interval) {
  xla::XlaOp one = xla::One(growth_tracker.builder(), xla::PrimitiveType::S32);
  xla::XlaOp one_float =
      xla::One(growth_tracker.builder(), xla::PrimitiveType::F32);
  xla::XlaOp found_inf =
      xla::ConvertElementType(found_inf_float, xla::PrimitiveType::PRED);
  const auto& growth_factor = XlaHelpers::ScalarValue<float>(
      scale_growth_factor,
      XlaHelpers::ShapeOfXlaOp(current_scale).element_type(),
      growth_tracker.builder());
  const auto& backoff_factor = XlaHelpers::ScalarValue<float>(
      scale_backoff_factor,
      XlaHelpers::ShapeOfXlaOp(current_scale).element_type(),
      growth_tracker.builder());
  const auto& growth_interval = XlaHelpers::ScalarValue<int>(
      scale_growth_interval,
      XlaHelpers::ShapeOfXlaOp(growth_tracker).element_type(),
      growth_tracker.builder());

  xla::XlaOp all_finite = xla::Not(found_inf);
  xla::XlaOp not_achieve_interval = xla::ConvertElementType(
      growth_interval - one - growth_tracker, xla::PrimitiveType::PRED);
  xla::XlaOp new_growth_tracker =
      (growth_tracker + one) *
      ConvertElementType(xla::And(all_finite, not_achieve_interval),
                         xla::PrimitiveType::S32);
  xla::XlaOp growth_factor_or_one = xla::Max(
      growth_factor * xla::ConvertElementType(
                          xla::And(all_finite, xla::Not(not_achieve_interval)),
                          xla::PrimitiveType::F32),
      one_float);
  xla::XlaOp backoff_factor_or_one =
      backoff_factor *
          xla::ConvertElementType(found_inf, xla::PrimitiveType::F32) +
      xla::ConvertElementType(all_finite, xla::PrimitiveType::F32);
  xla::XlaOp new_scale =
      current_scale * growth_factor_or_one * backoff_factor_or_one;
  return {new_growth_tracker, new_scale};
}

}  // namespace compiler
}  // namespace torch_lazy_tensors
