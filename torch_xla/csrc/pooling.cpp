#include "torch_xla/csrc/pooling.h"

#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/util.h>

#include "absl/status/status.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/util.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_lower_util.h"
#include "xla/client/lib/arithmetic.h"
#include "xla/client/lib/constants.h"
#include "xla/client/lib/pooling.h"
#include "xla/client/lib/slicing.h"
#include "xla/hlo/builder/lib/loops.h"

namespace torch_xla {
namespace {

const xla::PrimitiveType kIndicesType = xla::PrimitiveType::U32;

struct PoolingOpAttributes {
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
};

struct PoolSliceIndices {
  std::vector<xla::XlaOp> result_indices;
  std::vector<xla::XlaOp> input_indices;
};

struct InitValues {
  size_t append(xla::XlaOp op) {
    values.push_back(std::move(op));
    return values.size() - 1;
  }

  std::vector<xla::XlaOp> values;
};

xla::XlaComputation CreateGeComputation(xla::PrimitiveType type) {
  xla::XlaBuilder reduction_builder("xla_ge_computation");
  xla::XlaOp x = xla::Parameter(&reduction_builder, 0,
                                xla::ShapeUtil::MakeShape(type, {}), "x");
  xla::XlaOp y = xla::Parameter(&reduction_builder, 1,
                                xla::ShapeUtil::MakeShape(type, {}), "y");
  xla::Ge(x, y);
  return ConsumeValue(reduction_builder.Build());
}

xla::TensorFormat MakeNCHWFormat(int64_t spatial_dim_count) {
  return {/*batch_dimension=*/0,
          /*feature_dimension=*/1,
          /*spatial_dimensions=*/
          torch::lazy::Iota<int64_t>(spatial_dim_count, 2)};
}

// Construct the pooling attributes for the given kernel size, stride and
// padding.
PoolingOpAttributes MakePoolingOpAttributes(
    absl::Span<const int64_t> kernel_size_attr,
    absl::Span<const int64_t> stride_attr) {
  // Create a NCHW kernel size with 1 for batch size and feature.
  std::vector<int64_t> kernel_size(2, 1);
  kernel_size.insert(kernel_size.end(), kernel_size_attr.begin(),
                     kernel_size_attr.end());
  // Create a NCHW stride size with 1 for batch size and feature. Same as kernel
  // size if not specified.
  std::vector<int64_t> stride;
  if (stride_attr.empty()) {
    stride = kernel_size;
  } else {
    stride.resize(2, 1);
    stride.insert(stride.end(), stride_attr.begin(), stride_attr.end());
  }
  return {std::move(kernel_size), std::move(stride)};
}

// Compute the  pool kernel size required for the specified output_size
// from the given input_size, when the stride is the same as the kernel size.
std::vector<int64_t> AdaptivePoolKernelSize(
    absl::Span<const int64_t> input_size, absl::Span<const int64_t> output_size,
    int pool_dim) {
  // Create a NCHW kernel size with 1 for batch size and feature.
  std::vector<int64_t> kernel_size(2, 1);
  int64_t spatial_dim_off = input_size.size() - pool_dim;
  for (int spatial_dim = 0; spatial_dim < pool_dim; ++spatial_dim) {
    XLA_CHECK_EQ(
        input_size[spatial_dim_off + spatial_dim] % output_size[spatial_dim], 0)
        << "Target output size " << output_size[spatial_dim]
        << " doesn't divide the input size "
        << input_size[spatial_dim_off + spatial_dim];
    kernel_size.push_back(input_size[spatial_dim_off + spatial_dim] /
                          output_size[spatial_dim]);
  }
  return kernel_size;
}

struct BatchInput {
  xla::XlaOp batch_input;
  int64_t original_rank;
};

// Adds a batch dimension of size 1 if the input tensor doesn't have a batch
// dimension.
BatchInput CreateBatchInput(xla::XlaOp input, int64_t spatial_dim_count) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  int64_t rank = input_shape.rank();
  XLA_CHECK(rank == spatial_dim_count + 1 || rank == spatial_dim_count + 2)
      << "Input must be a " << spatial_dim_count + 1 << "-D or "
      << spatial_dim_count + 2 << "-D tensor";
  if (rank == spatial_dim_count + 1) {
    return {BuildUnsqueeze(input, 0), rank};
  }
  return {input, rank};
}

xla::XlaOp RemoveTrivialBatch(xla::XlaOp batch, int64_t original_rank,
                              int64_t spatial_dim_count) {
  if (original_rank == spatial_dim_count + 1) {
    return SqueezeTrivialDimension(batch, 0);
  }
  return batch;
}

// Creates low and high padding specification for the given padding (which is
// symmetric) and ceil mode. Additional high padding could be required when ceil
// mode is set.
std::vector<std::pair<int64_t, int64_t>> CeilModePadding(
    absl::Span<const int64_t> padding, const xla::Shape& input_shape,
    absl::Span<const int64_t> kernel_size, absl::Span<const int64_t> stride,
    bool ceil_mode, bool count_include_pad) {
  std::vector<std::pair<int64_t, int64_t>> ceil_mode_padding;
  for (int i = 0; i < padding.size(); ++i) {
    int64_t left_padding = padding[i];
    if (count_include_pad) {
      // if count_include_pad; the padding is added as XLA ops
      left_padding = 0;
    }
    int64_t input_size = input_shape.dimensions(2 + i);
    int64_t output_size_rem =
        (input_size + 2 * left_padding - kernel_size[i]) % stride[i];
    int64_t right_padding = left_padding;
    if (ceil_mode && output_size_rem != 0) {
      int64_t extra_padding = stride[i] - output_size_rem;
      int64_t new_output_size =
          (input_size + left_padding + right_padding + extra_padding -
           kernel_size[i] + stride[i] - 1) /
              stride[i] +
          1;
      // Ensure that the last pooling starts inside the image.
      int64_t size_to_compare = input_size + left_padding;
      if (count_include_pad) {
        // here left padding is reset to 0;
        // but input size already includes both left_padding and
        // right padding so we need to substract padding[i]
        size_to_compare = input_size - padding[i];
      }
      if ((new_output_size - 1) * stride[i] < size_to_compare) {
        right_padding += extra_padding;
      }
    }
    ceil_mode_padding.emplace_back(left_padding, right_padding);
  }
  return ceil_mode_padding;
}

xla::PaddingConfig MakeXlaPaddingConfig(absl::Span<const int64_t> padding) {
  xla::PaddingConfig padding_config;
  for (int i = 0; i < 2; ++i) {
    padding_config.add_dimensions();
  }
  for (int pad : padding) {
    xla::PaddingConfig::PaddingConfigDimension* dims =
        padding_config.add_dimensions();
    dims->set_edge_padding_low(pad);
    dims->set_edge_padding_high(pad);
  }
  return padding_config;
}

// Creates an XLA padding configuration from a padding attribute value.
xla::PaddingConfig MakeXlaPaddingConfig(
    std::vector<std::pair<int64_t, int64_t>> padding) {
  xla::PaddingConfig padding_config;
  for (int i = 0; i < 2; ++i) {
    padding_config.add_dimensions();
  }
  for (const auto& dim_padding : padding) {
    xla::PaddingConfig::PaddingConfigDimension* dims =
        padding_config.add_dimensions();
    dims->set_edge_padding_low(dim_padding.first);
    dims->set_edge_padding_high(dim_padding.second);
  }
  return padding_config;
}

xla::XlaOp CreatePoolIndicesIota(const xla::Shape& input_shape,
                                 xla::XlaBuilder* builder) {
  int64_t spatial_input_elements = 1;
  for (int64_t i = 2; i < input_shape.rank(); ++i) {
    spatial_input_elements *= input_shape.dimensions(i);
  }
  xla::XlaOp iota = xla::Iota(
      builder,
      xla::ShapeUtil::MakeShape(kIndicesType, {spatial_input_elements}), 0);
  xla::XlaOp reshaped_iota =
      xla::Reshape(iota, input_shape.dimensions().subspan(2));
  return xla::Broadcast(reshaped_iota, input_shape.dimensions().subspan(0, 2));
}

xla::XlaOp ComputeNoOverlapMaxPoolIndices(
    const xla::Shape& input_shape, xla::XlaOp padded_input,
    xla::XlaOp pool_result, const xla::PaddingConfig& padding_config,
    const PoolingOpAttributes& pooling_op_attributes) {
  const xla::Shape& padded_input_shape =
      ShapeHelper::ShapeOfXlaOp(padded_input);
  xla::XlaComputation select = xla::CreateScalarGeComputation(
      input_shape.element_type(), padded_input.builder());
  xla::XlaComputation scatter = xla::CreateScalarMaxComputation(
      input_shape.element_type(), padded_input.builder());
  xla::XlaOp init_value =
      xla::MinValue(padded_input.builder(), input_shape.element_type());
  xla::XlaOp scattered_pool = xla::SelectAndScatter(
      padded_input, select, pooling_op_attributes.kernel_size,
      pooling_op_attributes.stride, xla::Padding::kValid, pool_result,
      init_value, scatter);

  xla::XlaOp iota = CreatePoolIndicesIota(input_shape, padded_input.builder());
  xla::XlaOp invalid_iota_init =
      xla::MaxValue(padded_input.builder(), kIndicesType);
  xla::XlaOp padded_iota = xla::Pad(iota, invalid_iota_init, padding_config);
  xla::XlaOp invalid_iota =
      xla::Broadcast(invalid_iota_init, padded_input_shape.dimensions());
  xla::XlaOp scattered_indices = xla::Select(
      xla::Ne(scattered_pool, init_value), padded_iota, invalid_iota);
  xla::XlaComputation min_computation =
      xla::CreateScalarMinComputation(kIndicesType, padded_input.builder());
  return xla::ReduceWindow(scattered_indices, invalid_iota_init,
                           min_computation, pooling_op_attributes.kernel_size,
                           pooling_op_attributes.stride, xla::Padding::kValid);
}

PoolSliceIndices ComputeSliceIndices(xla::XlaOp linear_index,
                                     absl::Span<const int64_t> dimensions,
                                     absl::Span<const int64_t> window_strides) {
  xla::PrimitiveType scalar_type = XlaHelpers::TypeOfXlaOp(linear_index);
  std::vector<int64_t> strides = torch::lazy::ComputeArrayStrides(
      torch::lazy::ToVector<int64_t>(dimensions));
  PoolSliceIndices indices;
  xla::XlaOp current_index = linear_index;
  for (size_t i = 0; i < dimensions.size(); ++i) {
    xla::XlaOp dim_stride = XlaHelpers::ScalarValue(strides[i], scalar_type,
                                                    linear_index.builder());
    xla::XlaOp wnd_stride = XlaHelpers::ScalarValue(
        window_strides[i], scalar_type, linear_index.builder());
    indices.result_indices.push_back(current_index / dim_stride);
    indices.input_indices.push_back(wnd_stride * indices.result_indices.back());
    current_index = current_index % dim_stride;
  }
  return indices;
}

bool IsOverlapping(const PoolingOpAttributes& pooling_op_attributes) {
  XLA_CHECK_EQ(pooling_op_attributes.kernel_size.size(),
               pooling_op_attributes.stride.size());
  for (size_t i = 0; i < pooling_op_attributes.stride.size(); ++i) {
    if (pooling_op_attributes.stride[i] <
        pooling_op_attributes.kernel_size[i]) {
      return true;
    }
  }
  return false;
}

xla::XlaOp ComputeMaxPoolIndices(
    const xla::Shape& input_shape, xla::XlaOp padded_input,
    xla::XlaOp pool_result, const xla::PaddingConfig& padding_config,
    const PoolingOpAttributes& pooling_op_attributes) {
  if (!IsOverlapping(pooling_op_attributes)) {
    // The algorithm in ComputeNoOverlapMaxPoolIndices() only works if reduce
    // windows do not overlap. If they do, the reduce-window done on the indices
    // will find multiple indices within the window, and won't know what to
    // select. If XLA had a vardic reduce-window we could do that.
    return ComputeNoOverlapMaxPoolIndices(input_shape, padded_input,
                                          pool_result, padding_config,
                                          pooling_op_attributes);
  }

  // Slow version follows.
  // We loop through every window and compute the index. The slow code will only
  // be executed if the caller actually uses the indices, and only if the reduce
  // windows overlap.
  xla::XlaOp iota = CreatePoolIndicesIota(input_shape, padded_input.builder());
  xla::XlaOp padded_iota =
      xla::Pad(iota, xla::MaxValue(padded_input.builder(), kIndicesType),
               padding_config);

  const xla::Shape& pool_result_shape = ShapeHelper::ShapeOfXlaOp(pool_result);
  int64_t pool_elements = xla::ShapeUtil::ElementsIn(pool_result_shape);

  InitValues initial_values;
  size_t counter_id =
      initial_values.append(xla::Zero(padded_input.builder(), kIndicesType));
  size_t limit_id = initial_values.append(XlaHelpers::ScalarValue(
      pool_elements, kIndicesType, padded_input.builder()));
  size_t input_id = initial_values.append(padded_input);
  size_t pool_result_id = initial_values.append(pool_result);
  size_t iota_id = initial_values.append(padded_iota);
  size_t result_id = initial_values.append(
      xla::Zeros(padded_input.builder(),
                 xla::ShapeUtil::MakeShape(kIndicesType, {pool_elements})));

  auto cond_fn = [&](absl::Span<const xla::XlaOp> init,
                     xla::XlaBuilder* builder) -> absl::StatusOr<xla::XlaOp> {
    return xla::Lt(init[counter_id], init[limit_id]);
  };
  auto body_fn =
      [&](absl::Span<const xla::XlaOp> init,
          xla::XlaBuilder* builder) -> absl::StatusOr<std::vector<xla::XlaOp>> {
    PoolSliceIndices slice_indices =
        ComputeSliceIndices(init[counter_id], pool_result_shape.dimensions(),
                            pooling_op_attributes.stride);

    xla::XlaOp input_slice =
        xla::DynamicSlice(init[input_id], slice_indices.input_indices,
                          pooling_op_attributes.kernel_size);
    xla::XlaOp iota_slice =
        xla::DynamicSlice(init[iota_id], slice_indices.input_indices,
                          pooling_op_attributes.kernel_size);
    std::vector<int64_t> result_slice_sizes(
        pooling_op_attributes.kernel_size.size(), 1);
    xla::XlaOp pool_result_slice = xla::DynamicSlice(
        init[pool_result_id], slice_indices.result_indices, result_slice_sizes);

    xla::XlaComputation select =
        xla::CreateScalarGeComputation(input_shape.element_type(), builder);
    xla::XlaComputation scatter =
        xla::CreateScalarMaxComputation(input_shape.element_type(), builder);
    xla::XlaOp init_value = xla::MinValue(builder, input_shape.element_type());
    xla::XlaOp scattered_pool = xla::SelectAndScatter(
        input_slice, select, pooling_op_attributes.kernel_size,
        pooling_op_attributes.stride, xla::Padding::kValid, pool_result_slice,
        init_value, scatter);

    xla::XlaOp invalid_iota_init = xla::MaxValue(builder, kIndicesType);
    xla::XlaOp invalid_iota =
        xla::Broadcast(invalid_iota_init, pooling_op_attributes.kernel_size);
    xla::XlaOp scattered_indices = xla::Select(
        xla::Ne(scattered_pool, init_value), iota_slice, invalid_iota);
    xla::XlaComputation min_computation =
        xla::CreateScalarMinComputation(kIndicesType, builder);
    xla::XlaOp index =
        xla::ReduceWindow(scattered_indices, invalid_iota_init, min_computation,
                          pooling_op_attributes.kernel_size,
                          pooling_op_attributes.stride, xla::Padding::kValid);
    xla::XlaOp r1_index = xla::Reshape(index, {1});

    std::vector<xla::XlaOp> results(init.begin(), init.end());
    results[counter_id] = init[counter_id] + xla::One(builder, kIndicesType);
    results[result_id] = xla::DynamicUpdateSlice(results[result_id], r1_index,
                                                 {init[counter_id]});
    return results;
  };

  std::vector<xla::XlaOp> results = ConsumeValue(
      xla::WhileLoopHelper(cond_fn, body_fn, initial_values.values,
                           "ComputeMaxPoolIndices", padded_input.builder()));

  return xla::Reshape(results[result_id], pool_result_shape.dimensions());
}

}  // namespace

bool IsSupportedAdaptivePool(absl::Span<const int64_t> input_size,
                             absl::Span<const int64_t> output_size,
                             int pool_dim) {
  int64_t rank = input_size.size();
  XLA_CHECK_EQ(output_size.size(), pool_dim);
  for (int spatial_dim = 0; spatial_dim < pool_dim; ++spatial_dim) {
    if (input_size[rank - pool_dim + spatial_dim] % output_size[spatial_dim] !=
        0) {
      return false;
    }
  }
  return true;
}

MaxPoolResult BuildAdaptiveMaxPoolNd(xla::XlaOp input,
                                     absl::Span<const int64_t> output_size,
                                     int pool_dim) {
  BatchInput batch_input_info =
      CreateBatchInput(input, /*spatial_dim_count=*/pool_dim);
  const xla::Shape& input_shape =
      ShapeHelper::ShapeOfXlaOp(batch_input_info.batch_input);
  const auto kernel_size =
      AdaptivePoolKernelSize(input_shape.dimensions(), output_size, pool_dim);
  // AdaptiveMaxPool won't have padding.
  xla::PaddingConfig padding_config;
  for (int i = 0; i < kernel_size.size(); i++) {
    padding_config.add_dimensions();
  }
  xla::XlaOp batch_result = xla::MaxPool(
      /*operand=*/batch_input_info.batch_input,
      /*kernel_size=*/kernel_size,
      /*stride=*/kernel_size,
      /*padding=*/xla::Padding::kValid,
      /*data_format=*/MakeNCHWFormat(pool_dim));
  xla::XlaOp batch_indices = ComputeMaxPoolIndices(
      input_shape, batch_input_info.batch_input, batch_result, padding_config,
      {kernel_size, kernel_size});

  return {RemoveTrivialBatch(batch_result, batch_input_info.original_rank,
                             pool_dim),
          RemoveTrivialBatch(batch_indices, batch_input_info.original_rank,
                             pool_dim)};
}

xla::XlaOp BuildAdaptiveMaxPoolNdBackward(xla::XlaOp out_backprop,
                                          xla::XlaOp input, int pool_dim) {
  BatchInput batch_out_backprop_info =
      CreateBatchInput(/*input=*/out_backprop, /*spatial_dim_count=*/pool_dim);
  const xla::Shape& out_backprop_shape =
      ShapeHelper::ShapeOfXlaOp(batch_out_backprop_info.batch_input);
  XLA_CHECK_EQ(out_backprop_shape.rank(), pool_dim + 2)
      << "Invalid rank of gradient output";
  std::vector<int64_t> output_size(out_backprop_shape.dimensions().begin() + 2,
                                   out_backprop_shape.dimensions().end());
  xla::XlaBuilder* builder = out_backprop.builder();
  BatchInput batch_input_info = CreateBatchInput(input, pool_dim);
  const xla::Shape& input_shape =
      ShapeHelper::ShapeOfXlaOp(batch_input_info.batch_input);
  xla::XlaOp init_value = xla::Zero(builder, input_shape.element_type());
  xla::XlaComputation select = CreateGeComputation(input_shape.element_type());
  xla::XlaComputation scatter =
      XlaHelpers::CreateAddComputation(input_shape.element_type());
  const auto kernel_size =
      AdaptivePoolKernelSize(input_shape.dimensions(), output_size, pool_dim);
  // AdaptiveMaxPool won't have padding.
  std::vector<std::pair<int64_t, int64_t>> window_padding(2 + pool_dim, {0, 0});
  xla::XlaOp batch_result = xla::SelectAndScatterWithGeneralPadding(
      /*operand=*/batch_input_info.batch_input,
      /*select=*/select,
      /*window_dimensions=*/kernel_size,
      /*window_strides=*/kernel_size,
      /*padding=*/window_padding,
      /*source=*/batch_out_backprop_info.batch_input,
      /*init_value=*/init_value,
      /*scatter=*/scatter);
  return RemoveTrivialBatch(/*batch=*/batch_result,
                            /*original_rank=*/batch_input_info.original_rank,
                            /*spatial_dim_count=*/pool_dim);
}

MaxPoolResult BuildMaxPoolNd(xla::XlaOp input, int64_t spatial_dim_count,
                             absl::Span<const int64_t> kernel_size,
                             absl::Span<const int64_t> stride,
                             absl::Span<const int64_t> padding,
                             bool ceil_mode) {
  xla::XlaBuilder* builder = input.builder();
  BatchInput batch_input_info = CreateBatchInput(input, spatial_dim_count);
  const xla::Shape& input_shape =
      ShapeHelper::ShapeOfXlaOp(batch_input_info.batch_input);
  xla::XlaOp init_value = xla::MinValue(builder, input_shape.element_type());
  std::vector<std::pair<int64_t, int64_t>> ceil_padding = CeilModePadding(
      padding, input_shape, kernel_size, stride, ceil_mode, false);
  xla::PaddingConfig padding_config = MakeXlaPaddingConfig(ceil_padding);
  xla::XlaOp padded_input =
      xla::Pad(batch_input_info.batch_input, init_value, padding_config);
  PoolingOpAttributes pooling_op_attributes =
      MakePoolingOpAttributes(/*kernel_size_attr=*/kernel_size,
                              /*stride_attr=*/stride);
  xla::XlaOp batch_result = xla::MaxPool(
      /*operand=*/padded_input,
      /*kernel_size=*/pooling_op_attributes.kernel_size,
      /*stride=*/pooling_op_attributes.stride,
      /*padding=*/xla::Padding::kValid,
      /*data_format=*/MakeNCHWFormat(spatial_dim_count));
  xla::XlaOp batch_indices =
      ComputeMaxPoolIndices(input_shape, padded_input, batch_result,
                            padding_config, pooling_op_attributes);
  return {RemoveTrivialBatch(batch_result, batch_input_info.original_rank,
                             spatial_dim_count),
          RemoveTrivialBatch(batch_indices, batch_input_info.original_rank,
                             spatial_dim_count)};
}

xla::XlaOp BuildMaxPoolNdBackward(xla::XlaOp out_backprop, xla::XlaOp input,
                                  int64_t spatial_dim_count,
                                  absl::Span<const int64_t> kernel_size,
                                  absl::Span<const int64_t> stride,
                                  absl::Span<const int64_t> padding,
                                  bool ceil_mode) {
  xla::XlaBuilder* builder = out_backprop.builder();
  BatchInput batch_input_info = CreateBatchInput(input, spatial_dim_count);
  const xla::Shape& input_shape =
      ShapeHelper::ShapeOfXlaOp(batch_input_info.batch_input);
  xla::XlaOp init_value = xla::Zero(builder, input_shape.element_type());
  xla::XlaComputation select = CreateGeComputation(input_shape.element_type());
  xla::XlaComputation scatter =
      XlaHelpers::CreateAddComputation(input_shape.element_type());
  PoolingOpAttributes pooling_op_attributes =
      MakePoolingOpAttributes(/*kernel_size_attr=*/kernel_size,
                              /*stride_attr=*/stride);
  std::vector<std::pair<int64_t, int64_t>> window_padding;
  const auto ceil_mode_padding = CeilModePadding(
      padding, input_shape, kernel_size, stride, ceil_mode, false);
  window_padding.resize(2);
  window_padding.insert(window_padding.end(), ceil_mode_padding.begin(),
                        ceil_mode_padding.end());
  BatchInput batch_out_backprop_info =
      CreateBatchInput(out_backprop, spatial_dim_count);
  xla::XlaOp batch_result = xla::SelectAndScatterWithGeneralPadding(
      /*operand=*/batch_input_info.batch_input,
      /*select=*/select,
      /*window_dimensions=*/pooling_op_attributes.kernel_size,
      /*window_strides=*/pooling_op_attributes.stride,
      /*padding=*/window_padding,
      /*source=*/batch_out_backprop_info.batch_input,
      /*init_value=*/init_value,
      /*scatter=*/scatter);
  return RemoveTrivialBatch(/*batch=*/batch_result,
                            /*original_rank=*/batch_input_info.original_rank,
                            /*spatial_dim_count=*/spatial_dim_count);
}

xla::XlaOp BuildMaxUnpoolNd(const torch::lazy::BackendDevice& device,
                            xla::XlaOp input, xla::XlaOp indices,
                            absl::Span<const int64_t> output_size) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  XLA_CHECK_EQ(input_shape.rank(), 2 + output_size.size());

  xla::Shape zeros_shape = xla::ShapeUtil::MakeShape(
      input_shape.element_type(),
      {input_shape.dimensions(0), input_shape.dimensions(1),
       runtime::util::Multiply<int64_t>(output_size)});
  xla::XlaOp zeros = xla::Zeros(input.builder(), zeros_shape);
  xla::XlaOp init_value =
      xla::Broadcast(xla::MinValue(input.builder(), input_shape.element_type()),
                     zeros_shape.dimensions());
  xla::XlaOp flat_input =
      XlaHelpers::FlattenDimRange(input, 2, output_size.size());
  xla::XlaOp flat_indices =
      XlaHelpers::FlattenDimRange(indices, 2, output_size.size());

  auto combiner_fn = [](xla::XlaOp x, xla::XlaOp y) -> xla::XlaOp {
    return xla::Max(x, y);
  };
  ScatterOptions options(combiner_fn);
  options.indices_are_unique = false;
  options.init_value =
      xla::MinValue(input.builder(), input_shape.element_type());
  xla::XlaOp scatter_result =
      CreateScatter(device, init_value, flat_indices, flat_input,
                    /*dim=*/2, options);
  xla::XlaOp result =
      xla::Select(xla::Ne(scatter_result, init_value), scatter_result, zeros);

  std::vector<int64_t> result_sizes(
      {input_shape.dimensions(0), input_shape.dimensions(1)});
  result_sizes.insert(result_sizes.end(), output_size.begin(),
                      output_size.end());
  return XlaHelpers::DynamicReshape(result, result_sizes);
}

xla::XlaOp BuildMaxUnpoolNdBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                    xla::XlaOp indices,
                                    absl::Span<const int64_t> output_size) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  xla::XlaOp flat_grad_output =
      XlaHelpers::FlattenDimRange(grad_output, 2, output_size.size());
  xla::XlaOp flat_indices =
      XlaHelpers::FlattenDimRange(indices, 2, output_size.size());
  xla::XlaOp gather_result = xla::TorchGather(
      flat_grad_output, flat_indices, /*dim=*/2,
      IsSparseGather(flat_grad_output, flat_indices, /*dim=*/2));

  return XlaHelpers::DynamicReshapeAs(gather_result, input_shape);
}

xla::XlaOp BuildAvgPoolNd(xla::XlaOp input, int64_t spatial_dim_count,
                          absl::Span<const int64_t> kernel_size,
                          absl::Span<const int64_t> stride,
                          absl::Span<const int64_t> padding, bool ceil_mode,
                          bool count_include_pad) {
  PoolingOpAttributes pooling_op_attributes =
      MakePoolingOpAttributes(/*kernel_size_attr=*/kernel_size,
                              /*stride_attr=*/stride);
  BatchInput batch_input_info = CreateBatchInput(input, spatial_dim_count);
  const xla::Shape& input_shape =
      ShapeHelper::ShapeOfXlaOp(batch_input_info.batch_input);

  if (count_include_pad) {
    xla::PaddingConfig padding_config = MakeXlaPaddingConfig(padding);
    auto dtype = ShapeHelper::ShapeOfXlaOp(input).element_type();
    auto padding_value = XlaHelpers::ScalarValue(0, dtype, input.builder());
    batch_input_info.batch_input =
        xla::Pad(batch_input_info.batch_input, padding_value, padding_config);
  }

  const auto ceil_mode_padding = CeilModePadding(
      padding, ShapeHelper::ShapeOfXlaOp(batch_input_info.batch_input),
      kernel_size, stride, ceil_mode, count_include_pad);

  xla::XlaOp batch_result = xla::AvgPool(
      /*operand=*/batch_input_info.batch_input,
      /*kernel_size=*/pooling_op_attributes.kernel_size,
      /*stride=*/pooling_op_attributes.stride,
      /*padding=*/ceil_mode_padding,
      /*data_format=*/MakeNCHWFormat(spatial_dim_count),
      /*counts_include_padding=*/false);  // already compensated in XLA

  return RemoveTrivialBatch(/*batch=*/batch_result,
                            /*original_rank=*/batch_input_info.original_rank,
                            /*spatial_dim_count=*/spatial_dim_count);
}

xla::XlaOp BuildAvgPoolNdBackward(xla::XlaOp out_backprop, xla::XlaOp input,
                                  int64_t spatial_dim_count,
                                  absl::Span<const int64_t> kernel_size,
                                  absl::Span<const int64_t> stride,
                                  absl::Span<const int64_t> padding,
                                  bool ceil_mode, bool count_include_pad) {
  PoolingOpAttributes pooling_op_attributes =
      MakePoolingOpAttributes(/*kernel_size_attr=*/kernel_size,
                              /*stride_attr=*/stride);
  BatchInput batch_input_info = CreateBatchInput(input, spatial_dim_count);
  const xla::Shape& gradients_shape =
      ShapeHelper::ShapeOfXlaOp(batch_input_info.batch_input);
  const auto ceil_mode_padding = CeilModePadding(
      padding, gradients_shape, kernel_size, stride, ceil_mode, false);
  BatchInput batch_out_backprop_info =
      CreateBatchInput(out_backprop, spatial_dim_count);
  xla::XlaOp batch_result = xla::AvgPoolGrad(
      /*out_backprop=*/batch_out_backprop_info.batch_input,
      /*gradients_size=*/gradients_shape.dimensions(),
      /*kernel_size=*/pooling_op_attributes.kernel_size,
      /*stride=*/pooling_op_attributes.stride,
      /*spatial_padding=*/ceil_mode_padding,
      /*data_format=*/MakeNCHWFormat(spatial_dim_count),
      /*counts_include_padding=*/count_include_pad);
  return RemoveTrivialBatch(/*batch=*/batch_result,
                            /*original_rank=*/batch_input_info.original_rank,
                            /*spatial_dim_count=*/spatial_dim_count);
}

xla::XlaOp BuildAdaptiveAvgPool(xla::XlaOp input,
                                absl::Span<const int64_t> output_size,
                                int pool_dim) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  const auto kernel_size =
      AdaptivePoolKernelSize(input_shape.dimensions(), output_size, pool_dim);
  std::vector<std::pair<int64_t, int64_t>> no_padding(pool_dim);
  BatchInput batch_input_info =
      CreateBatchInput(input, /*spatial_dim_count=*/pool_dim);
  xla::XlaOp batch_result = xla::AvgPool(
      /*operand=*/batch_input_info.batch_input,
      /*kernel_size=*/kernel_size,
      /*stride=*/kernel_size,
      /*padding=*/no_padding,
      /*data_format=*/MakeNCHWFormat(pool_dim),
      /*counts_include_padding=*/false);
  return RemoveTrivialBatch(/*batch=*/batch_result,
                            /*original_rank=*/batch_input_info.original_rank,
                            /*spatial_dim_count=*/pool_dim);
}

xla::XlaOp BuildAdaptiveAvgPool3d(xla::XlaOp input,
                                  absl::Span<const int64_t> output_size) {
  XLA_CHECK_EQ(output_size.size(), 3) << "Invalid output size rank";
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  XLA_CHECK(input_shape.rank() == 4 || input_shape.rank() == 5)
      << "Only 4D or 5D tensors supported";
  return BuildAdaptiveAvgPool(input, output_size, /*pool_dim=*/3);
}

xla::XlaOp BuildAdaptiveAvgPool2d(xla::XlaOp input,
                                  absl::Span<const int64_t> output_size) {
  XLA_CHECK_EQ(output_size.size(), 2) << "Invalid output size rank";
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  XLA_CHECK(input_shape.rank() == 4 || input_shape.rank() == 3)
      << "Only 4D or 3D tensors supported";
  return BuildAdaptiveAvgPool(input, output_size, /*pool_dim=*/2);
}

xla::XlaOp BuildAdaptiveAvgPoolBackward(xla::XlaOp out_backprop,
                                        xla::XlaOp input, int pool_dim) {
  BatchInput batch_out_backprop_info =
      CreateBatchInput(/*input=*/out_backprop, /*spatial_dim_count=*/pool_dim);
  const xla::Shape& out_backprop_shape =
      ShapeHelper::ShapeOfXlaOp(batch_out_backprop_info.batch_input);
  XLA_CHECK_EQ(out_backprop_shape.rank(), pool_dim + 2)
      << "Invalid rank of gradient output";
  std::vector<int64_t> output_size(out_backprop_shape.dimensions().begin() + 2,
                                   out_backprop_shape.dimensions().end());
  auto gradients_size = XlaHelpers::SizesOfXlaOp(input);
  if (gradients_size.size() == pool_dim + 1) {
    gradients_size.insert(gradients_size.begin(), 1);
  }
  const auto kernel_size =
      AdaptivePoolKernelSize(gradients_size, output_size, pool_dim);
  std::vector<std::pair<int64_t, int64_t>> no_padding(pool_dim);
  xla::XlaOp batch_result = xla::AvgPoolGrad(
      /*out_backprop=*/batch_out_backprop_info.batch_input,
      /*gradients_size=*/gradients_size,
      /*kernel_size=*/kernel_size,
      /*stride=*/kernel_size,
      /*spatial_padding=*/no_padding,
      /*data_format=*/MakeNCHWFormat(pool_dim),
      /*counts_include_padding=*/false);
  return RemoveTrivialBatch(
      /*batch=*/batch_result,
      /*original_rank=*/batch_out_backprop_info.original_rank,
      /*spatial_dim_count=*/pool_dim);
}

xla::XlaOp BuildAdaptiveAvgPool3dBackward(xla::XlaOp out_backprop,
                                          xla::XlaOp input) {
  const xla::Shape& gradients_shape = ShapeHelper::ShapeOfXlaOp(input);
  XLA_CHECK(gradients_shape.rank() == 4 || gradients_shape.rank() == 5)
      << "Only 4D or 5D tensors supported";
  return BuildAdaptiveAvgPoolBackward(out_backprop, input,
                                      /*pool_dim=*/3);
}

xla::XlaOp BuildAdaptiveAvgPool2dBackward(xla::XlaOp out_backprop,
                                          xla::XlaOp input) {
  const xla::Shape& gradients_shape = ShapeHelper::ShapeOfXlaOp(input);
  XLA_CHECK(gradients_shape.rank() == 4 || gradients_shape.rank() == 3)
      << "Only 4D or 3D tensors supported";
  return BuildAdaptiveAvgPoolBackward(out_backprop, input,
                                      /*pool_dim=*/2);
}

}  // namespace torch_xla
