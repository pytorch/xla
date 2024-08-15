#include "torch_xla/csrc/convolution_helper.h"

#include "absl/status/status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/tensor_float_32_utils.h"
#include "xla/client/xla_builder.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_xla {

// -------------Convolution Helper Function Start-------------------------
// Convolution helper functions below are copied/inspired from TF2XLA bridge
// This part of helpers are origionally from
// https://github.com/tensorflow/tensorflow/blob/7f47eaf439d2b81de1aa24b10ed57eabd519dbdb/tensorflow/core/util/tensor_format.cc

// Convert a TensorFormat into string.
std::string ToString(TensorFormat format) {
  switch (format) {
    case FORMAT_NHWC:
      return "NHWC";
    case FORMAT_NCHW:
      return "NCHW";
    case FORMAT_NCHW_VECT_C:
      return "NCHW_VECT_C";
    case FORMAT_NHWC_VECT_W:
      return "NHWC_VECT_W";
    case FORMAT_HWNC:
      return "HWNC";
    case FORMAT_HWCN:
      return "HWCN";
    default:
      LOG(FATAL) << "Invalid Format: " << static_cast<tsl::int32>(format);
      return "INVALID_FORMAT";
  }
}

// This part of helpers are origionally from
// https://github.com/tensorflow/tensorflow/blob/7f47eaf439d2b81de1aa24b10ed57eabd519dbdb/tensorflow/compiler/tf2xla/kernels/conv_op_helpers.cc

// Performs some basic checks on ConvOpAttrs that are true for all kinds of
// XLA convolutions (as currently implemented).
absl::Status CheckConvAttrs(const ConvOpAttrs& attrs) {
  const int num_dims = attrs.num_spatial_dims + 2;
  const int attrs_strides_size = attrs.strides.size();
  if (attrs_strides_size != num_dims) {
    return tsl::errors::InvalidArgument(
        "Sliding window strides field must specify ", num_dims, " dimensions");
  }
  int batch_dim = GetTensorBatchDimIndex(num_dims, attrs.data_format);
  int feature_dim = GetTensorFeatureDimIndex(num_dims, attrs.data_format);
  if (attrs.strides[batch_dim] != 1 || attrs.strides[feature_dim] != 1) {
    return tsl::errors::Unimplemented(
        "Current implementation does not yet support strides in the batch and "
        "depth dimensions.");
  }
  const int attrs_dilations_size = attrs.dilations.size();
  if (attrs_dilations_size != num_dims) {
    return tsl::errors::InvalidArgument("Dilations field must specify ",
                                        num_dims, " dimensions");
  }
  if (attrs.dilations[batch_dim] != 1 || attrs.dilations[feature_dim] != 1) {
    return tsl::errors::Unimplemented(
        "Current implementation does not support dilations in the batch and "
        "depth dimensions.");
  }
  for (int i = 0; i < attrs.num_spatial_dims; ++i) {
    int input_dim = GetTensorSpatialDimIndex(num_dims, attrs.data_format, i);
    if (attrs.dilations[input_dim] < 1) {
      return tsl::errors::Unimplemented("Dilation values must be positive; ", i,
                                        "th spatial dimension had dilation ",
                                        attrs.dilations[input_dim]);
    }
  }
  return tsl::OkStatus();
}

// Returns the expanded size of a filter used for depthwise convolution.
// If `shape` is [H, W, ..., M, N] returns [H, W, ..., 1, M*N].
xla::Shape GroupedFilterShapeForDepthwiseConvolution(
    const xla::Shape& filter_shape) {
  int64_t input_feature_dim = filter_shape.dimensions_size() - 2;
  int64_t output_feature_dim = filter_shape.dimensions_size() - 1;
  int64_t depthwise_multiplier = filter_shape.dimensions(output_feature_dim);
  int64_t input_feature = filter_shape.dimensions(input_feature_dim);

  // Create a [H, W, ..., 1, M*N] reshape of the filter.
  xla::Shape grouped_filter_shape = filter_shape;
  grouped_filter_shape.set_dimensions(input_feature_dim, 1);
  grouped_filter_shape.set_dimensions(output_feature_dim,
                                      depthwise_multiplier * input_feature);
  return grouped_filter_shape;
}

// This part of helpers are origionally from
// https://github.com/tensorflow/tensorflow/blob/7f39a389d5b82d6aca13240c21f2647c3ebdb765/tensorflow/core/framework/kernel_shape_util.cc

absl::Status GetWindowedOutputSizeVerboseV2(
    int64_t input_size, int64_t filter_size, int64_t dilation_rate,
    int64_t stride, Padding padding_type, int64_t* output_size,
    int64_t* padding_before, int64_t* padding_after) {
  if (stride <= 0) {
    return tsl::errors::InvalidArgument("Stride must be > 0, but got ", stride);
  }
  if (dilation_rate < 1) {
    return tsl::errors::InvalidArgument("Dilation rate must be >= 1, but got ",
                                        dilation_rate);
  }

  // See also the parallel implementation in GetWindowedOutputSizeFromDimsV2.
  int64_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  switch (padding_type) {
    case Padding::VALID:
      *output_size = (input_size - effective_filter_size + stride) / stride;
      *padding_before = *padding_after = 0;
      break;
    case Padding::EXPLICIT:
      *output_size = (input_size + *padding_before + *padding_after -
                      effective_filter_size + stride) /
                     stride;
      break;
    case Padding::SAME:
      *output_size = (input_size + stride - 1) / stride;
      const int64_t padding_needed =
          std::max(int64_t{0}, (*output_size - 1) * stride +
                                   effective_filter_size - input_size);
      // For odd values of total padding, add more padding at the 'right'
      // side of the given dimension.
      *padding_before = padding_needed / 2;
      *padding_after = padding_needed - *padding_before;
      break;
  }
  if (*output_size < 0) {
    return tsl::errors::InvalidArgument(
        "Computed output size would be negative: ", *output_size,
        " [input_size: ", input_size,
        ", effective_filter_size: ", effective_filter_size,
        ", stride: ", stride, "]");
  }
  return tsl::OkStatus();
}

// This part of helpers are origionally from
// https://github.com/tensorflow/tensorflow/blob/7f39a389d5b82d6aca13240c21f2647c3ebdb765/tensorflow/core/kernels/conv_grad_shape_utils.cc

// Check dimension
tsl::Status ConvBackpropExtractAndVerifyDimension(
    std::string_view label, const xla::Shape& input_shape,
    const xla::Shape& filter_shape, const xla::Shape& output_shape,
    const absl::Span<const tsl::int32> dilations,
    const std::vector<tsl::int32>& strides, Padding padding,
    int64_t padding_before, int64_t padding_after, int spatial_dim,
    int filter_spatial_dim, ConvBackpropSpatialDimension* dim) {
  dim->input_size = input_shape.dimensions(spatial_dim);
  dim->filter_size = filter_shape.dimensions(filter_spatial_dim);
  dim->output_size = output_shape.dimensions(spatial_dim);
  dim->stride = strides[spatial_dim];
  dim->dilation = dilations[spatial_dim];
  int64_t out_size = 0;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
      dim->input_size, dim->filter_size, dim->dilation, dim->stride, padding,
      &out_size, &padding_before, &padding_after));
  if (dim->output_size != out_size) {
    return tsl::errors::InvalidArgument(
        label, ": Size of out_backprop doesn't match computed: ", "actual = ",
        dim->output_size, ", computed = ", out_size,
        " spatial_dim: ", spatial_dim, " input: ", dim->input_size,
        " filter: ", dim->filter_size, " output: ", dim->output_size,
        " stride: ", dim->stride, " dilation: ", dim->dilation);
  }

  int64_t effective_filter_size = (dim->filter_size - 1) * dim->dilation + 1;
  dim->expanded_output_size = (dim->output_size - 1) * dim->stride + 1;
  const auto padded_out_size = dim->input_size + effective_filter_size - 1;
  dim->pad_before = effective_filter_size - 1 - padding_before;
  dim->pad_after =
      padded_out_size - dim->expanded_output_size - dim->pad_before;
  VLOG(2) << label << ": expanded_out = " << dim->expanded_output_size
          << ", effective_filter_size = " << effective_filter_size
          << ", padded_out = " << padded_out_size
          << ", pad_before = " << dim->pad_before
          << ", pad_after = " << dim->pad_after
          << ", dilation = " << dim->dilation << ", strides = " << dim->stride;
  return tsl::OkStatus();
}

// Check dimension
tsl::Status ConvBackpropComputeDimensionsV2(
    std::string_view label, int num_spatial_dims, const xla::Shape& input_shape,
    const xla::Shape& filter_shape, const xla::Shape& out_backprop_shape,
    absl::Span<const tsl::int32> dilations,
    const std::vector<tsl::int32>& strides, Padding padding,
    TensorFormat data_format, ConvBackpropDimensions* dims,
    absl::Span<const int64_t> explicit_paddings) {
  // The + 2 in the following line is for the batch and feature dimensions.
  const int num_dims = num_spatial_dims + 2;
  if (input_shape.rank() != num_dims) {
    return tsl::errors::InvalidArgument(label, ": input must be ", num_dims,
                                        "-dimensional");
  }
  if (filter_shape.rank() != num_dims) {
    return tsl::errors::InvalidArgument(label, ": filter must be ", num_dims,
                                        "-dimensional");
  }
  if (out_backprop_shape.rank() != num_dims) {
    return tsl::errors::InvalidArgument(label, ": out_backprop must be ",
                                        num_dims, "-dimensional");
  }
  int batch_dim = GetTensorBatchDimIndex(num_dims, data_format);
  dims->batch_size = input_shape.dimensions(batch_dim);
  if (dims->batch_size != out_backprop_shape.dimensions(batch_dim)) {
    return tsl::errors::InvalidArgument(
        label, ": input and out_backprop must have the same batch size.",
        " Input batch: ", dims->batch_size,
        ", outbackprop batch: ", out_backprop_shape.dimensions(batch_dim),
        ", batch_dim: ", batch_dim);
  }

  int feature_dim = GetTensorFeatureDimIndex(num_dims, data_format);
  dims->in_depth = input_shape.dimensions(feature_dim);
  // The input and output feature dimensions are the second last and last
  // dimensions of the filter Tensor.
  VLOG(2) << "input vs filter_in depth " << dims->in_depth << " "
          << filter_shape.dimensions(num_dims - 2);
  if (filter_shape.dimensions(num_dims - 2) <= 0) {
    return tsl::errors ::InvalidArgument(
        label, ": filter depth must be strictly greated than zero");
  }
  if (dims->in_depth % filter_shape.dimensions(num_dims - 2)) {
    return tsl::errors::InvalidArgument(
        label, ": input depth must be evenly divisible by filter depth");
  }
  dims->out_depth = filter_shape.dimensions(num_dims - 1);
  if (dims->out_depth != out_backprop_shape.dimensions(feature_dim)) {
    return tsl::errors::InvalidArgument(
        label, ": filter and out_backprop must have the same out_depth");
  }
  dims->spatial_dims.resize(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    int image_dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
    int64_t padding_before = -1, padding_after = -1;
    if (padding == Padding::EXPLICIT) {
      padding_before = explicit_paddings[2 * image_dim];
      padding_after = explicit_paddings[2 * image_dim + 1];
    }
    TF_RETURN_IF_ERROR(ConvBackpropExtractAndVerifyDimension(
        label, input_shape, filter_shape, out_backprop_shape, dilations,
        strides, padding, padding_before, padding_after, image_dim, i,
        &dims->spatial_dims[i]));
  }
  return tsl::OkStatus();
}

// This part of helpers are origionally from
// https://github.com/tensorflow/tensorflow/blob/7f47eaf439d2b81de1aa24b10ed57eabd519dbdb/tensorflow/compiler/tf2xla/kernels/conv_op_helpers.cc

xla::PrecisionConfig GetPrecisionConfig() {
  xla::PrecisionConfig::Precision precision =
      tsl::tensor_float_32_execution_enabled() ? xla::PrecisionConfig::DEFAULT
                                               : xla::PrecisionConfig::HIGHEST;
  xla::PrecisionConfig config;
  const int num_inputs = 2;
  config.mutable_operand_precision()->Reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    config.add_operand_precision(precision);
  }
  return config;
}

// Returns the transposed filter for use in BackpropInput of group convolution.
xla::XlaOp TransposeFilterForGroupConvolutionBackpropInput(
    xla::XlaOp filter, const xla::Shape& filter_shape, int64_t num_groups,
    int num_spatial_dims) {
  // 1. Reshape from [H, W, ..., filter_in_depth, out_depth] to [H, W, ...,
  // filter_in_depth, G, out_depth / G]
  int num_dims = filter_shape.dimensions_size();
  CHECK_GE(num_dims, 2);  // Crash OK
  xla::Shape new_shape = filter_shape;
  new_shape.set_dimensions(num_dims - 1, num_groups);
  new_shape.add_dimensions(filter_shape.dimensions(num_dims - 1) / num_groups);
  xla::XlaOp result = xla::Reshape(filter, new_shape.dimensions());

  // 2. Transpose to [H, W, ..., G, filter_in_depth, out_depth / G]
  std::vector<int64_t> transpose_dims(num_dims + 1);
  std::iota(transpose_dims.begin(), transpose_dims.end(), 0);
  std::swap(transpose_dims[num_spatial_dims],
            transpose_dims[num_spatial_dims + 1]);
  result = xla::Transpose(result, transpose_dims);

  // 3. Reshape to [H, W, ..., in_depth, out_depth / G]
  result = xla::Collapse(result, {num_spatial_dims, num_spatial_dims + 1});
  return result;
}

// Wrapper for ConvGeneralDilated and check dim.
tsl::StatusOr<xla::XlaOp> MakeXlaBackpropInputConvOp(
    std::string_view type_string, const xla::Shape& input_shape,
    xla::XlaOp filter, xla::XlaOp out_backprop, const ConvOpAttrs& attrs,
    xla::XlaOp* input_sizes) {
  TF_RETURN_IF_ERROR(CheckConvAttrs(attrs));

  int num_dims = attrs.num_spatial_dims + 2;
  int batch_dim = GetTensorBatchDimIndex(num_dims, attrs.data_format);
  int feature_dim = GetTensorFeatureDimIndex(num_dims, attrs.data_format);

  auto* builder = filter.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape filter_shape, builder->GetShape(filter));
  TF_ASSIGN_OR_RETURN(xla::Shape out_backprop_shape,
                      builder->GetShape(out_backprop));

  int64_t in_depth = input_shape.dimensions(feature_dim),
          filter_in_depth = filter_shape.dimensions(attrs.num_spatial_dims),
          feature_group_count =
              attrs.depthwise ? filter_in_depth : in_depth / filter_in_depth;

  xla::Shape grouped_filter_shape =
      attrs.depthwise ? GroupedFilterShapeForDepthwiseConvolution(filter_shape)
                      : filter_shape;

  // Dimension check
  ConvBackpropDimensions dims;
  TF_RETURN_IF_ERROR(ConvBackpropComputeDimensionsV2(
      type_string, attrs.num_spatial_dims, input_shape, grouped_filter_shape,
      out_backprop_shape, attrs.dilations, attrs.strides, attrs.padding,
      attrs.data_format, &dims, attrs.explicit_paddings));

  // The input gradients are computed by a convolution of the output
  // gradients and the filter, with some appropriate padding. See the
  // comment at the top of conv_grad_shape_utils.h(TF) for details.
  xla::ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(batch_dim);
  dnums.set_output_batch_dimension(batch_dim);
  dnums.set_input_feature_dimension(feature_dim);
  dnums.set_output_feature_dimension(feature_dim);

  // TF filter shape is [ H, W, ..., inC, outC ]
  // Transpose the input and output features for computing the gradient.
  dnums.set_kernel_input_feature_dimension(attrs.num_spatial_dims + 1);
  dnums.set_kernel_output_feature_dimension(attrs.num_spatial_dims);

  std::vector<int64_t> kernel_spatial_dims(attrs.num_spatial_dims);
  std::vector<std::pair<int64_t, int64_t>> padding(attrs.num_spatial_dims);
  std::vector<int64_t> lhs_dilation(attrs.num_spatial_dims);
  std::vector<int64_t> rhs_dilation(attrs.num_spatial_dims);
  std::vector<int64_t> ones(attrs.num_spatial_dims, 1);
  xla::PaddingType padding_type = xla::PaddingType::PADDING_INVALID;
  for (int i = 0; i < attrs.num_spatial_dims; ++i) {
    int64_t dim = GetTensorSpatialDimIndex(num_dims, attrs.data_format, i);
    if (out_backprop_shape.is_dynamic_dimension(dim)) {
      TF_RET_CHECK(attrs.padding == Padding::VALID ||
                   attrs.padding == Padding::SAME)
          << "Dynamic convolution only supports valid and same padding";
      if (attrs.padding == Padding::VALID) {
        padding_type = xla::PaddingType::PADDING_VALID;
      }
      if (attrs.padding == Padding::SAME) {
        padding_type = xla::PaddingType::PADDING_SAME;
      }
    }
    dnums.add_input_spatial_dimensions(dim);
    dnums.add_kernel_spatial_dimensions(i);
    dnums.add_output_spatial_dimensions(dim);

    kernel_spatial_dims[i] = i;
    padding[i] = {dims.spatial_dims[i].pad_before,
                  dims.spatial_dims[i].pad_after};
    lhs_dilation[i] = dims.spatial_dims[i].stride;
    rhs_dilation[i] = attrs.dilations[dim];
  }
  xla::PrecisionConfig precision_config = GetPrecisionConfig();

  if (feature_group_count != 1 && !attrs.depthwise) {
    filter = TransposeFilterForGroupConvolutionBackpropInput(
        filter, filter_shape, feature_group_count, attrs.num_spatial_dims);
  }
  // Mirror the filter in the spatial dimensions.
  filter = xla::Rev(filter, kernel_spatial_dims);
  if (padding_type != xla::PaddingType::PADDING_INVALID) {
    TF_RET_CHECK(input_sizes != nullptr);
    return xla::DynamicConvInputGrad(
        *input_sizes, out_backprop, filter, /*window_strides=*/ones, padding,
        lhs_dilation, rhs_dilation, dnums,
        /*feature_group_count=*/
        feature_group_count,
        /*batch_group_count=*/1, &precision_config, padding_type);
  }
  // activation gradients
  //   = gradients (with padding and dilation) <conv> mirrored_weights
  return xla::ConvGeneralDilated(out_backprop, filter, /*window_strides=*/ones,
                                 padding, lhs_dilation, rhs_dilation, dnums,
                                 /*feature_group_count=*/
                                 feature_group_count,
                                 /*batch_group_count=*/1, &precision_config);
}

// Wrapper for ConvGeneralDilated and check dim.
tsl::StatusOr<xla::XlaOp> MakeXlaBackpropFilterConvOp(
    std::string_view type_string, xla::XlaOp activations,
    const xla::Shape& filter_shape, xla::XlaOp gradients,
    const ConvOpAttrs& attrs) {
  TF_RETURN_IF_ERROR(CheckConvAttrs(attrs));

  auto* builder = activations.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape activations_shape,
                      builder->GetShape(activations));
  TF_ASSIGN_OR_RETURN(xla::Shape out_backprop_shape,
                      builder->GetShape(gradients));
  xla::XlaOp filter_backprop;

  xla::Shape input_shape = activations_shape;
  xla::Shape output_shape = out_backprop_shape;

  const xla::Shape grouped_filter_shape =
      attrs.depthwise ? GroupedFilterShapeForDepthwiseConvolution(filter_shape)
                      : filter_shape;
  // Reuse dimension computation logic from conv_grad_shape_utils.cc.
  ConvBackpropDimensions dims;
  // The filter gradients are computed by a convolution of the input
  // activations and the output gradients, with some appropriate padding.
  // See the comment at the top of conv_grad_shape_utils.h(TF) for details.
  xla::ConvolutionDimensionNumbers dnums;

  // Dimension check
  TF_RETURN_IF_ERROR(ConvBackpropComputeDimensionsV2(
      type_string, attrs.num_spatial_dims, activations_shape,
      grouped_filter_shape, out_backprop_shape, attrs.dilations, attrs.strides,
      attrs.padding, attrs.data_format, &dims, attrs.explicit_paddings));

  // Obtain some useful dimensions:
  // The last two dimensions of the filter are the input and output shapes.
  int num_dims = attrs.num_spatial_dims + 2;
  int n_dim = GetTensorBatchDimIndex(num_dims, attrs.data_format);
  int c_dim = GetTensorFeatureDimIndex(num_dims, attrs.data_format);
  int64_t in_depth = input_shape.dimensions(c_dim),
          filter_in_depth = filter_shape.dimensions(attrs.num_spatial_dims),
          batch_group_count =
              attrs.depthwise ? filter_in_depth : in_depth / filter_in_depth;

  std::vector<std::pair<int64_t, int64_t>> padding(attrs.num_spatial_dims);
  std::vector<int64_t> rhs_dilation(attrs.num_spatial_dims);
  std::vector<int64_t> window_strides(attrs.num_spatial_dims);
  std::vector<int64_t> ones(attrs.num_spatial_dims, 1);

  // Swap n_dim and c_dim in the activations.
  dnums.set_input_batch_dimension(c_dim);
  dnums.set_input_feature_dimension(n_dim);

  // The gradients become the RHS of the convolution.
  // The gradients have shape [batch, out_rows, out_cols, ..., out_depth]
  // where the batch becomes the input feature for the convolution.
  dnums.set_kernel_input_feature_dimension(n_dim);
  dnums.set_kernel_output_feature_dimension(c_dim);

  dnums.set_output_batch_dimension(attrs.num_spatial_dims);
  dnums.set_output_feature_dimension(attrs.num_spatial_dims + 1);

  // Tensorflow filter shape is [ H, W, ..., inC, outC ].
  for (int i = 0; i < attrs.num_spatial_dims; ++i) {
    dnums.add_output_spatial_dimensions(i);
  }
  xla::PaddingType padding_type = xla::PaddingType::PADDING_INVALID;
  for (int64_t i = 0; i < attrs.num_spatial_dims; ++i) {
    int64_t dim = GetTensorSpatialDimIndex(num_dims, attrs.data_format, i);
    if (activations_shape.is_dynamic_dimension(dim)) {
      TF_RET_CHECK(attrs.padding == Padding::VALID ||
                   attrs.padding == Padding::SAME)
          << "Dynamic convolution only supports valid and same padding";
      if (attrs.padding == Padding::VALID) {
        padding_type = xla::PaddingType::PADDING_VALID;
      }
      if (attrs.padding == Padding::SAME) {
        padding_type = xla::PaddingType::PADDING_SAME;
      }
    }
    dnums.add_input_spatial_dimensions(dim);
    dnums.add_kernel_spatial_dimensions(dim);
    rhs_dilation[i] = dims.spatial_dims[i].stride;
    window_strides[i] = attrs.dilations[dim];

    // We will also need to pad the input with zeros such that after the
    // convolution, we get the right size for the filter.
    // The padded_in_rows should be such that when we convolve this with the
    // expanded_out_rows as a filter, we should get filter_rows back.

    const int64_t padded_in_size =
        dims.spatial_dims[i].expanded_output_size +
        (dims.spatial_dims[i].filter_size - 1) * attrs.dilations[dim];

    // However it can be smaller than input_rows: in this
    // case it means some of the inputs are not used.
    //
    // An example is to have input_cols = 3, filter_cols = 2 and stride = 2:
    //
    // INPUT =  [ A  B  C ]
    //
    // FILTER = [ x y ]
    //
    // and the output will only have one column: a = A * x + B * y
    //
    // and input "C" is not used at all.
    //
    // We apply negative padding in this case.
    const int64_t pad_total = padded_in_size - dims.spatial_dims[i].input_size;

    // + For the EXPLICIT padding, we pad the top/left side with the explicit
    //   padding and pad the bottom/right side with the remaining space.
    // + For the VALID padding, we don't pad anything on the top/left side
    //   and pad the bottom/right side with the remaining space.
    // + For the SAME padding, we pad top/left side the same as bottom/right
    //   side.
    //
    // In addition, if the padded input size is smaller than the input size,
    // we need to ignore some training elements of the input. We do this by
    // applying negative padding on the right/bottom.
    const int64_t pad_before =
        attrs.padding == Padding::EXPLICIT ? attrs.explicit_paddings[2 * dim]
        : attrs.padding == Padding::SAME   ? std::max<int64_t>(pad_total / 2, 0)
                                           : 0;
    padding[i] = {pad_before, pad_total - pad_before};
  }
  xla::PrecisionConfig precision_config = GetPrecisionConfig();

  // Besides padding the input, we will also expand output_rows to
  //    expanded_out_rows = (output_rows - 1) * stride + 1
  // with zeros in between:
  //
  //      a . . . b . . . c . . . d . . . e
  //
  // This is done by specifying the window dilation factors in the
  // convolution HLO below.
  if (padding_type != xla::PaddingType::PADDING_INVALID) {
    filter_backprop = xla::DynamicConvKernelGrad(
        activations, gradients, window_strides, padding, /*lhs_dilation=*/ones,
        rhs_dilation, dnums,
        /*feature_group_count=*/1,
        /*batch_group_count=*/batch_group_count, &precision_config,
        padding_type);
  } else {
    filter_backprop = xla::ConvGeneralDilated(
        activations, gradients, window_strides, padding, /*lhs_dilation=*/ones,
        rhs_dilation, dnums,
        /*feature_group_count=*/1,
        /*batch_group_count=*/batch_group_count, &precision_config);
  }

  if (attrs.depthwise) {
    filter_backprop = xla::Reshape(filter_backprop, filter_shape.dimensions());
  }

  return filter_backprop;
}

// -------------Convolution Helper Function End--------------------------

}  // namespace torch_xla
