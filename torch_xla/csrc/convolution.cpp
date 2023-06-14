#include "torch_xla/csrc/convolution.h"

#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h" // CheckConvAttrs // MakeXlaBackpropInputConvOp // ConvAttrs
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/xla_lower_util.h"
#include "tensorflow/core/util/tensor_format.h" // GetTensorBatchDimIndex // GetTensorFeatureDimIndex // GetTensorSpatialDimIndex
#include "tensorflow/tsl/platform/errors.h" // tsl::errors::InvalidArgument // 
#include "tensorflow/core/kernels/conv_grad_shape_utils.h" // ConvBackpropDimensions // 
#include "tensorflow/core/util/padding.h" // tensorflow::Padding // 
#include "tensorflow/core/util/tensor_format.h" // TensorFormat
#include "tensorflow/core/framework/tensor_shape.h" // TensorShape
#include "tensorflow/compiler/tf2xla/shape_util.h" // XLAShapeToTensorShape
#include "tensorflow/core/kernels/conv_grad_shape_utils.h" // ConvBackpropComputeDimensionsV2
#include "tensorflow/compiler/xla/xla_data.pb.h" // ConvolutionDimensionNumbers // PaddingType // PrecisionConfig
#include "tensorflow/compiler/xla/client/xla_builder.h" // DynamicConvInputGrad // ConvGeneralDilated
#include "tensorflow/tsl/platform/tensor_float_32_utils.h" // tensor_float_32_execution_enabled

namespace torch_xla {
namespace {
// clang-format off
/* Lower PyTorch Conv & its backward to XLA
 * This file covers lowerings of both forward and backward of conv op.
 *   - BuildConvolutionOverrideable
 *     - BuildTransposedConvolution
 *     - Normal(non-transpose) convolution
 *   - BuildConvolutionBackwardOverrideable
 *     - BuildConvBackwardInput
 *     - BuildConvBackwardWeight
 *     - BuildGradBias
 *
 * Here're detailed steps from a 4D PyTorch inputs to inputs calling into
 * ConvGeneralDilated (the most general conv op in XLA).
 * PyTorch input format & shape:
 *   - input: [N, Cin, Hin, Win]
 *   - weight: [Cout, Cin / groups, Hker, Wker]
 *   - output: [N, Cout, Hout, Wout]
 *
 * Output, grad_input and grad_weight can all be calculated as
 * ConvGeneralDilated with proper transpose/padding etc.
 *   - output: conv(input, weight)
 *   - grad_input: conv(grad_output, weight^T) (with padding etc)
 *   - grad_weight: conv(input^T, grad_output)
 *
 * XLA provides the following wrappers instead of calling into raw
 * ConvGeneralDilated.
 * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/tf2xla/kernels/conv_op_helpers.cc
 *   - MakeXlaForwardConvOp (not used in our lowering, see below)
 *   - MakeXlaBackpropInputConvOp
 *   - MakeXlaBackpropFilterConvOp
 *
 * Lowering for non group convolution is straightforward, in this note
 * we focus on grouped conv and depthwise conv which need a bit special handling.
 * Shapes in the section below use format in XLA implementation [N, H, W, C]
 * instead of PyTorch convention [N, C, H, W].
 * Here are the shapes that feed into ConvGeneralDilated ops, note these
 * are different from input of wrappers like MakeXlaBackpropInputConvOp.
 * We try to use these shapes to explain what happens from a PyTorch tensor
 * to call into conv op.
 *
 * For group(non-depthwise) convolutions(G > 1, Cout = M * G, M = channel_multiplier):
 * forward: (conv with groups = G)
 *   - input: [N, Hin, Win, Cin]
 *   - filter: [Hker, Wker, Cin / G, Cout]
 * grad_input: (conv with groups = G)
 *   - input: [N, Hout, Wout, Cout]
 *   - filter: [Hker, Wker, Cout, Cin / G] // func: TransposeFilterForGroupConvolutionBackpropInput
 *          => [Hker, Wker, Cout / G, Cin]
 * grad_weight: (conv with groups = G)
 *   - input: [N, Hin, Win, Cin] // func: TransposeInputForGroupConvolutionBackpropFilter
 *         => [G * N, Hin, Win, Cin / G] // swap batch & channel dimension
 *         => [Cin / G, Hin, Win, G * N]
 *   - filter: [N, Hout, Wout, Cout] // func: FilterTransposePermutation
 *          => [Hout, Wout, Cout, N] // swap batch & channel dimension
 *          => [Hout, Wout, N, Cout]
 *
 * For depthwise conv (Cin = G, Cout = M * G, M = channel_multiplier):
 * This case is special since XLA expects a filter of shape [Hker, Wker, Cin, M]
 * and it reshapes it to [Hker, Wker, 1, Cout] back and forth inside the wrapper
 * functions.
 *
 * Since PyTorch already gives the weight in shape [Hker, Wker, 1, Cout] in
 * depthwise convolution, there's no need to do additional reshapes to match to
 * XLA expected format. This is also why we use raw ConvGeneralDilated instead
 * of MakeXlaForwardConvOp in forward graph. For code simplicity we still want
 * to use the MakeXlaBackpropInputConvOp and MakeXlaBackpropFilterConvOp given
 * they have many useful steps that we don't want to duplicate here, we simply
 * enforce depthwise = false inside those functions, so that we skip the reshape
 * steps XLA has with a [Hker, Wker, Cin, M] input.
 *
 * forward: (conv with groups = G)
 *   - input: [N, Hin, Win, Cin]
 *   - filter: [Hker, Wker, 1, Cout]
 * grad_input: (conv with groups = G)
 *   - input: [N, Hout, Wout, Cout]
 *   - filter: [Hker, Wker, Cout, 1] // func: TransposeFilterForGroupConvolutionBackpropInput
 *          => [Hker, Wker, Cout / Cin, Cin]
 * grad_weight: (conv with groups = G)
 *   - input: [N, Hin, Win, Cin]
 *         => [G * N, Hin, Win, 1] // swap batch & channel dimension
 *         => [1, Hin, Win, G * N]
 *   - filter: [N, Hout, Wout, Cout] // func: FilterTransposePermutation
 *          => [Hout, Wout, Cout, N] // swap batch & channel dimension
 *          => [Hout, Wout, N, Cout]
 */
// clang-format on

// Input needs to be NCHW format.
xla::XlaOp PadInputFromOutputSize(xla::XlaOp input,
                                  absl::Span<const int64_t> stride,
                                  absl::Span<const int64_t> output_padding,
                                  bool unpad = false) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  int64_t num_spatial = input_shape.rank() - 2;
  // No padding for batch dimension and features dimension.
  std::vector<int64_t> expected_input_sizes{input_shape.dimensions(0),
                                            input_shape.dimensions(1)};
  for (int64_t spatial_dim = 0; spatial_dim < num_spatial; ++spatial_dim) {
    int64_t input_size = input_shape.dimensions(2 + spatial_dim);
    // Input_size needs to increase by pad_to_input to generate the output
    // that includes output_padding. The formula is derived from the output size
    // calculation in the BuildTransposedConvolution.
    int64_t pad_to_input =
        ((input_size - 1) * stride[spatial_dim] + output_padding[spatial_dim]) /
            stride[spatial_dim] +
        1 - input_size;
    expected_input_sizes.push_back(input_size +
                                   (unpad ? -pad_to_input : pad_to_input));
  }
  return PadToSize(input, expected_input_sizes);
}

// Create a TF convolution metadata structure out of PyTorch convolution
// attributes.
tensorflow::ConvOpAttrs MakeConvOpAttrs(
    absl::Span<const int64_t> spatial_stride,
    absl::Span<const int64_t> spatial_padding,
    absl::Span<const int64_t> spatial_dilation, bool depthwise) {
  int num_spatial_dims = spatial_stride.size();
  XLA_CHECK_EQ(spatial_padding.size(), num_spatial_dims);
  XLA_CHECK_EQ(spatial_dilation.size(), num_spatial_dims);
  tensorflow::ConvOpAttrs conv_op_attrs;
  conv_op_attrs.depthwise = depthwise;
  conv_op_attrs.num_spatial_dims = num_spatial_dims;
  // Stride, dilation and padding must be set for the batch and feature in the
  // TF convolution metadata. Set them to 1 (stride and dilation) or 0 (padding)
  // for the batch and feature dimensions.
  conv_op_attrs.dilations = {1, 1};
  std::copy(spatial_dilation.begin(), spatial_dilation.end(),
            std::back_inserter(conv_op_attrs.dilations));
  conv_op_attrs.strides = {1, 1};
  std::copy(spatial_stride.begin(), spatial_stride.end(),
            std::back_inserter(conv_op_attrs.strides));
  conv_op_attrs.padding = tensorflow::Padding::EXPLICIT;
  // https://github.com/tensorflow/tensorflow/blob/ec81825aaf7e848d9f8ddffdf1e0d20aebe9172c/tensorflow/core/util/padding.cc#L40
  // explicit_padding requires to have (spatial_dims + 2) * 2 elements
  conv_op_attrs.explicit_paddings.resize(4);
  for (int spatial_dim = 0; spatial_dim < num_spatial_dims; ++spatial_dim) {
    conv_op_attrs.explicit_paddings.push_back(spatial_padding[spatial_dim]);
    conv_op_attrs.explicit_paddings.push_back(spatial_padding[spatial_dim]);
  }
  conv_op_attrs.data_format = tensorflow::TensorFormat::FORMAT_NCHW;
  return conv_op_attrs;
}

// Transpose filter shape to have [channel, batch] as last two dimensions.
// 4D case: (N, C, H, W) -> (H, W, C, N)
const std::vector<int64_t>& FilterTransposePermutation(const int64_t k) {
  if (k == 4) {
    static std::vector<int64_t>* permutation =
        new std::vector<int64_t>({2, 3, 1, 0});
    return *permutation;
  } else if (k == 5) {
    static std::vector<int64_t>* permutation =
        new std::vector<int64_t>({2, 3, 4, 1, 0});
    return *permutation;
  } else {
    XLA_ERROR() << "Invalid rank: " << k;
  }
}

// Bias broadcast based on output shape produces:
// (N, H, W) + (C,) = (N, H, W, C)
// This permutation does (N, H, W, C) -> (N, C, H, W)
const std::vector<int64_t>& BiasTransposePermutation(const int64_t k) {
  if (k == 4) {
    static std::vector<int64_t>* permutation =
        new std::vector<int64_t>({0, 3, 1, 2});
    return *permutation;
  } else if (k == 5) {
    static std::vector<int64_t>* permutation =
        new std::vector<int64_t>({0, 4, 1, 2, 3});
    return *permutation;
  } else {
    XLA_ERROR() << "Invalid rank: " << k;
  }
}

// Reduce bias from (N, C, H, W) to (C,)
const std::vector<int64_t>& BiasReduceDimensions(const int64_t k) {
  if (k == 4) {
    static std::vector<int64_t>* reduce_dim =
        new std::vector<int64_t>({0, 2, 3});
    return *reduce_dim;
  } else if (k == 5) {
    static std::vector<int64_t>* reduce_dim =
        new std::vector<int64_t>({0, 2, 3, 4});
    return *reduce_dim;
  } else {
    XLA_ERROR() << "Invalid rank: " << k;
  }
}

std::vector<std::pair<int64_t, int64_t>> MakePadding(
    absl::Span<const int64_t> padding) {
  std::vector<std::pair<int64_t, int64_t>> dims_padding;
  for (const auto dim_padding : padding) {
    dims_padding.emplace_back(dim_padding, dim_padding);
  }
  return dims_padding;
}

// Performs some basic checks on ConvOpAttrs that are true for all kinds of XLA
// convolutions (as currently implemented).
tsl::Status PTXLACheckConvAttrs(const tensorflow::ConvOpAttrs& attrs) {
  const int num_dims = attrs.num_spatial_dims + 2;
  const int attrs_strides_size = attrs.strides.size();
  if (attrs_strides_size != num_dims) {
    return tsl::errors::InvalidArgument("Sliding window strides field must specify ",
                                   num_dims, " dimensions");
  }
  int batch_dim = tensorflow::GetTensorBatchDimIndex(num_dims, attrs.data_format);
  int feature_dim = tensorflow::GetTensorFeatureDimIndex(num_dims, attrs.data_format);
  if (attrs.strides[batch_dim] != 1 || attrs.strides[feature_dim] != 1) {
    return tsl::errors::Unimplemented(
        "Current implementation does not yet support strides in the batch and "
        "depth dimensions.");
  }
  const int attrs_dilations_size = attrs.dilations.size();
  if (attrs_dilations_size != num_dims) {
    return tsl::errors::InvalidArgument("Dilations field must specify ", num_dims,
                                   " dimensions");
  }
  if (attrs.dilations[batch_dim] != 1 || attrs.dilations[feature_dim] != 1) {
    return tsl::errors::Unimplemented(
        "Current implementation does not support dilations in the batch and "
        "depth dimensions.");
  }
  for (int i = 0; i < attrs.num_spatial_dims; ++i) {
    int input_dim = tensorflow::GetTensorSpatialDimIndex(num_dims, attrs.data_format, i);
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
xla::Shape PTXLAGroupedFilterShapeForDepthwiseConvolution(
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

// Wrapper around ConvBackpropComputeDimensions that converts from XLA shapes
// to TensorShapes.
tsl::Status PTXLAConvBackpropComputeDimensionsV2XlaShapes(
    tsl::StringPiece label, int num_spatial_dims, const xla::Shape& input_shape,
    const xla::Shape& filter_shape, const xla::Shape& out_backprop_shape,
    absl::Span<const tsl::int32> dilations, const std::vector<tsl::int32>& strides,
    tensorflow::Padding padding, tensorflow::TensorFormat data_format, tensorflow::ConvBackpropDimensions* dims,
    absl::Span<const int64_t> explicit_paddings) {
  tensorflow::TensorShape input_tensor_shape, filter_tensor_shape,
      out_backprop_tensor_shape;
  TF_RETURN_IF_ERROR(tensorflow::XLAShapeToTensorShape(input_shape, &input_tensor_shape));
  TF_RETURN_IF_ERROR(tensorflow::XLAShapeToTensorShape(filter_shape, &filter_tensor_shape));
  TF_RETURN_IF_ERROR(
      tensorflow::XLAShapeToTensorShape(out_backprop_shape, &out_backprop_tensor_shape));
  return tensorflow::ConvBackpropComputeDimensionsV2(
      label, num_spatial_dims, input_tensor_shape, filter_tensor_shape,
      out_backprop_tensor_shape, dilations, strides, padding, explicit_paddings,
      data_format, dims);
}

xla::PrecisionConfig PTXLAGetPrecisionConfig() {
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
xla::XlaOp PTXLATransposeFilterForGroupConvolutionBackpropInput(
    xla::XlaOp filter, const xla::Shape& filter_shape, int64_t num_groups,
    int num_spatial_dims) {
  // 1. Reshape from [H, W, ..., filter_in_depth, out_depth] to [H, W, ...,
  // filter_in_depth, G, out_depth / G]
  int num_dims = filter_shape.dimensions_size();
  CHECK_GE(num_dims, 2);  // Crash OK
  xla::Shape new_shape = filter_shape;
  new_shape.set_dimensions(num_dims - 1, num_groups);
  new_shape.add_dimensions(filter_shape.dimensions(num_dims - 1) / num_groups);
  xla::XlaOp result = xla::Reshape(filter, new_shape.dimensions()); // TODO

  // 2. Transpose to [H, W, ..., G, filter_in_depth, out_depth / G]
  std::vector<int64_t> transpose_dims(num_dims + 1);
  std::iota(transpose_dims.begin(), transpose_dims.end(), 0);
  std::swap(transpose_dims[num_spatial_dims],
            transpose_dims[num_spatial_dims + 1]);
  result = xla::Transpose(result, transpose_dims); // TODO

  // 3. Reshape to [H, W, ..., in_depth, out_depth / G]
  result = xla::Collapse(result, {num_spatial_dims, num_spatial_dims + 1}); // TODO
  return result;
}

tsl::StatusOr<xla::XlaOp> PTXLAMakeXlaBackpropInputConvOp(tsl::StringPiece type_string,
                                                const xla::Shape& input_shape,
                                                xla::XlaOp filter,
                                                xla::XlaOp out_backprop,
                                                const tensorflow::ConvOpAttrs& attrs,
                                                xla::XlaOp* input_sizes = nullptr) {
  TF_RETURN_IF_ERROR(PTXLACheckConvAttrs(attrs));

  int num_dims = attrs.num_spatial_dims + 2;
  int batch_dim = tensorflow::GetTensorBatchDimIndex(num_dims, attrs.data_format);
  int feature_dim = tensorflow::GetTensorFeatureDimIndex(num_dims, attrs.data_format);

  auto* builder = filter.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape filter_shape, builder->GetShape(filter));
  TF_ASSIGN_OR_RETURN(xla::Shape out_backprop_shape,
                      builder->GetShape(out_backprop));

  int64_t in_depth = input_shape.dimensions(feature_dim),
          filter_in_depth = filter_shape.dimensions(attrs.num_spatial_dims),
          feature_group_count =
              attrs.depthwise ? filter_in_depth : in_depth / filter_in_depth;

  xla::Shape grouped_filter_shape =
      attrs.depthwise ? PTXLAGroupedFilterShapeForDepthwiseConvolution(filter_shape)
                      : filter_shape;
  // Reuse dimension computation logic from conv_grad_shape_utils.cc.
  tensorflow::ConvBackpropDimensions dims;
  TF_RETURN_IF_ERROR(PTXLAConvBackpropComputeDimensionsV2XlaShapes(
      type_string, attrs.num_spatial_dims, input_shape, grouped_filter_shape,
      out_backprop_shape, attrs.dilations, attrs.strides, attrs.padding,
      attrs.data_format, &dims, attrs.explicit_paddings));

  // The input gradients are computed by a convolution of the output
  // gradients and the filter, with some appropriate padding. See the
  // comment at the top of conv_grad_shape_utils.h for details.

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
    int64_t dim = tensorflow::GetTensorSpatialDimIndex(num_dims, attrs.data_format, i);
    if (out_backprop_shape.is_dynamic_dimension(dim)) {
      TF_RET_CHECK(attrs.padding == tensorflow::VALID || attrs.padding == tensorflow::SAME)
          << "Dynamic convolution only supports valid and same padding";
      if (attrs.padding == tensorflow::VALID) {
        padding_type = xla::PaddingType::PADDING_VALID;
      }
      if (attrs.padding == tensorflow::SAME) {
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
  xla::PrecisionConfig precision_config = PTXLAGetPrecisionConfig();

  if (feature_group_count != 1 && !attrs.depthwise) {
    filter = PTXLATransposeFilterForGroupConvolutionBackpropInput(
        filter, filter_shape, feature_group_count, attrs.num_spatial_dims);
  }
  // Mirror the filter in the spatial dimensions.
  filter = xla::Rev(filter, kernel_spatial_dims); // TODO
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

// Computes the input gradient for a convolution.
xla::XlaOp BuildConvBackwardInput(xla::XlaOp grad_output, xla::XlaOp kernel,
                                  const xla::Shape& input_shape,
                                  absl::Span<const int64_t> spatial_stride,
                                  absl::Span<const int64_t> spatial_padding,
                                  absl::Span<const int64_t> spatial_dilation,
                                  int64_t groups) {
  tensorflow::ConvOpAttrs conv_op_attrs =
      MakeConvOpAttrs(spatial_stride, spatial_padding, spatial_dilation, false);
  xla::XlaOp kernel_transposed =
      xla::Transpose(kernel, FilterTransposePermutation(input_shape.rank()));
  return ConsumeValue(PTXLAMakeXlaBackpropInputConvOp(
      "conv_backward_input", input_shape, kernel_transposed, grad_output,
      conv_op_attrs));
}

// Computes the kernel gradient for a convolution.
xla::XlaOp BuildConvBackwardWeight(xla::XlaOp grad_output, xla::XlaOp input,
                                   const xla::Shape& kernel_shape,
                                   absl::Span<const int64_t> spatial_stride,
                                   absl::Span<const int64_t> spatial_padding,
                                   absl::Span<const int64_t> spatial_dilation,
                                   int64_t groups) {
  tensorflow::ConvOpAttrs conv_op_attrs =
      MakeConvOpAttrs(spatial_stride, spatial_padding, spatial_dilation, false);
  auto transpose_permutation = FilterTransposePermutation(kernel_shape.rank());
  auto inv_transpose_permutation =
      xla::InversePermutation(transpose_permutation);
  xla::Shape transposed_weight_shape =
      xla::ShapeUtil::PermuteDimensions(transpose_permutation, kernel_shape);
  xla::XlaOp conv = ConsumeValue(tensorflow::MakeXlaBackpropFilterConvOp(
      "conv_backward_weight", input, transposed_weight_shape, grad_output,
      conv_op_attrs));

  // Reorder the dimensions of the filter gradient to match the NCHW convention
  // of PyTorch. The original result of the convolution has the spatial and
  // feature dimensions swapped and the spatial dimensions reversed.
  return xla::Transpose(conv, inv_transpose_permutation);
}

xla::XlaOp BuildGradBias(xla::XlaOp grad_output) {
  const xla::Shape& grad_output_shape = ShapeHelper::ShapeOfXlaOp(grad_output);
  // The bias contribution is linear in each output feature. Reduce the
  // remaining dimensions to get a tensor of the same shape as the bias, rank-1
  // with number of output features elements.
  return xla::Reduce(
      grad_output,
      xla::Zero(grad_output.builder(), grad_output_shape.element_type()),
      XlaHelpers::CreateAddComputation(grad_output_shape.element_type()),
      BiasReduceDimensions(grad_output_shape.rank()));
}

xla::XlaOp BuildTransposedConvolution(xla::XlaOp input, xla::XlaOp kernel,
                                      absl::Span<const int64_t> stride,
                                      absl::Span<const int64_t> padding,
                                      absl::Span<const int64_t> dilation,
                                      absl::Span<const int64_t> output_padding,
                                      int64_t groups) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  const xla::Shape& kernel_shape = ShapeHelper::ShapeOfXlaOp(kernel);
  int64_t num_spatial = input_shape.rank() - 2;
  // We only support 2D or 3D convolution.
  XLA_CHECK(num_spatial == 2 || num_spatial == 3) << num_spatial;
  // Fold group into output_size feature dimension
  int64_t features_size = kernel_shape.dimensions(1) * groups;
  std::vector<int64_t> output_size{input_shape.dimensions(0), features_size};
  for (int spatial_dim = 0; spatial_dim < num_spatial; ++spatial_dim) {
    output_size.push_back(
        (input_shape.dimensions(2 + spatial_dim) - 1) * stride[spatial_dim] -
        2 * padding[spatial_dim] +
        dilation[spatial_dim] * (kernel_shape.dimensions(2 + spatial_dim) - 1) +
        output_padding[spatial_dim] + 1);
  }
  // Pad the input to match the output_padding added to the output size.
  xla::XlaOp padded_input =
      PadInputFromOutputSize(input, stride, output_padding);
  return BuildConvBackwardInput(
      padded_input, kernel,
      xla::ShapeUtil::MakeShape(input_shape.element_type(), output_size),
      stride, padding, dilation, /*groups=*/1);
}

ConvGrads BuildTransposedConvolutionBackward(
    xla::XlaOp grad_output, xla::XlaOp input, xla::XlaOp kernel,
    absl::Span<const int64_t> stride, absl::Span<const int64_t> padding,
    absl::Span<const int64_t> dilation,
    absl::Span<const int64_t> output_padding, int64_t groups) {
  // grad_output includes output_padding, hence we need to pad the input and
  // unpad the grad_input.
  xla::XlaOp grad_input =
      BuildConvolutionOverrideable(grad_output, kernel, stride, padding,
                                   dilation, false, output_padding, groups);
  xla::XlaOp unpadded_grad_input = PadInputFromOutputSize(
      grad_input, stride, output_padding, /*unpad=*/true);
  xla::XlaOp padded_input =
      PadInputFromOutputSize(input, stride, output_padding);
  xla::XlaOp grad_weight = BuildConvBackwardWeight(
      padded_input, grad_output, ShapeHelper::ShapeOfXlaOp(kernel), stride,
      padding, dilation, groups);
  xla::XlaOp grad_bias = BuildGradBias(grad_output);
  return {unpadded_grad_input, grad_weight, grad_bias};
}

}  // namespace

xla::XlaOp BuildConvolutionOverrideable(
    xla::XlaOp input, xla::XlaOp kernel, absl::Span<const int64_t> stride,
    absl::Span<const int64_t> padding, absl::Span<const int64_t> dilation,
    bool transposed, absl::Span<const int64_t> output_padding, int64_t groups) {
  if (transposed) {
    return BuildTransposedConvolution(input, kernel, stride, padding, dilation,
                                      output_padding, groups);
  } else {
    auto dims_padding = MakePadding(padding);
    xla::PrecisionConfig precision_config =
        XlaHelpers::BuildPrecisionConfig(XlaHelpers::mat_mul_precision());
    return xla::ConvGeneralDilated(
        input, kernel, stride, dims_padding,
        /*lhs_dilation*/ {},
        /*rhs_dilation*/ dilation,
        /*dimension_numbers*/
        xla::XlaBuilder::CreateDefaultConvDimensionNumbers(stride.size()),
        /*feature_group_count*/ groups,
        /*batch_group_count=*/1, &precision_config);
  }
}

xla::XlaOp BuildConvolutionOverrideableBias(
    xla::XlaOp input, xla::XlaOp kernel, xla::XlaOp bias,
    absl::Span<const int64_t> stride, absl::Span<const int64_t> padding,
    absl::Span<const int64_t> dilation, bool transposed,
    absl::Span<const int64_t> output_padding, int64_t groups) {
  xla::XlaOp conv =
      BuildConvolutionOverrideable(input, kernel, stride, padding, dilation,
                                   transposed, output_padding, groups);
  auto broadcast_sizes = XlaHelpers::SizesOfXlaOp(conv);
  // Remove the channels dimension.
  broadcast_sizes.erase(broadcast_sizes.begin() + 1);
  // Make the bias match the output dimensions.
  xla::XlaOp bias_broadcast =
      xla::Transpose(xla::Broadcast(bias, broadcast_sizes),
                     BiasTransposePermutation(broadcast_sizes.size() + 1));
  return conv + bias_broadcast;
}

ConvGrads BuildConvolutionBackwardOverrideable(
    xla::XlaOp grad_output, xla::XlaOp input, xla::XlaOp kernel,
    absl::Span<const int64_t> stride, absl::Span<const int64_t> padding,
    absl::Span<const int64_t> dilation, bool transposed,
    absl::Span<const int64_t> output_padding, int64_t groups) {
  if (transposed) {
    return BuildTransposedConvolutionBackward(grad_output, input, kernel,
                                              stride, padding, dilation,
                                              output_padding, groups);
  } else {
    xla::XlaOp grad_input = BuildConvBackwardInput(
        grad_output, kernel, ShapeHelper::ShapeOfXlaOp(input), stride, padding,
        dilation, groups);
    xla::XlaOp grad_weight = BuildConvBackwardWeight(
        grad_output, input, ShapeHelper::ShapeOfXlaOp(kernel), stride, padding,
        dilation, groups);
    xla::XlaOp grad_bias = BuildGradBias(grad_output);
    return {grad_input, grad_weight, grad_bias};
  }
}
}  // namespace torch_xla
