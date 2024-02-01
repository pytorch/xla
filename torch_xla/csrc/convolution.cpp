#include "torch_xla/csrc/convolution.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/xla_lower_util.h"
#include "xla/client/lib/constants.h"

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
 * Below helpers are inspired by TF2XLA implementation of the Convolution
 * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/tf2xla/kernels/conv_op_helpers.cc
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
 * to use the MakeXlaBackpropInputConvOp and MakeXlaBackpropFilterConvOp,
 * we simply enforce depthwise = false inside those functions, so that we skip the
 * reshape steps XLA has with a [Hker, Wker, Cin, M] input.
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

// Create ConvOpAttrs
ConvOpAttrs MakeConvOpAttrs(absl::Span<const int64_t> spatial_stride,
                            absl::Span<const int64_t> spatial_padding,
                            absl::Span<const int64_t> spatial_dilation,
                            bool depthwise) {
  int num_spatial_dims = spatial_stride.size();
  XLA_CHECK_EQ(spatial_padding.size(), num_spatial_dims);
  XLA_CHECK_EQ(spatial_dilation.size(), num_spatial_dims);
  ConvOpAttrs conv_op_attrs;
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
  conv_op_attrs.padding = Padding::EXPLICIT;
  // https://github.com/tensorflow/tensorflow/blob/ec81825aaf7e848d9f8ddffdf1e0d20aebe9172c/tensorflow/core/util/padding.cc#L40
  // explicit_padding requires to have (spatial_dims + 2) * 2 elements
  conv_op_attrs.explicit_paddings.resize(4);
  for (int spatial_dim = 0; spatial_dim < num_spatial_dims; ++spatial_dim) {
    conv_op_attrs.explicit_paddings.push_back(spatial_padding[spatial_dim]);
    conv_op_attrs.explicit_paddings.push_back(spatial_padding[spatial_dim]);
  }
  conv_op_attrs.data_format = TensorFormat::FORMAT_NCHW;
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

// Computes the input gradient for a convolution.
xla::XlaOp BuildConvBackwardInput(xla::XlaOp grad_output, xla::XlaOp kernel,
                                  const xla::Shape& input_shape,
                                  absl::Span<const int64_t> spatial_stride,
                                  absl::Span<const int64_t> spatial_padding,
                                  absl::Span<const int64_t> spatial_dilation,
                                  int64_t groups) {
  ConvOpAttrs conv_op_attrs =
      MakeConvOpAttrs(spatial_stride, spatial_padding, spatial_dilation, false);
  xla::XlaOp kernel_transposed =
      xla::Transpose(kernel, FilterTransposePermutation(input_shape.rank()));
  return ConsumeValue(MakeXlaBackpropInputConvOp("conv_backward_input",
                                                 input_shape, kernel_transposed,
                                                 grad_output, conv_op_attrs));
}

// Computes the kernel gradient for a convolution.
xla::XlaOp BuildConvBackwardWeight(xla::XlaOp grad_output, xla::XlaOp input,
                                   const xla::Shape& kernel_shape,
                                   absl::Span<const int64_t> spatial_stride,
                                   absl::Span<const int64_t> spatial_padding,
                                   absl::Span<const int64_t> spatial_dilation,
                                   int64_t groups) {
  ConvOpAttrs conv_op_attrs =
      MakeConvOpAttrs(spatial_stride, spatial_padding, spatial_dilation, false);
  auto transpose_permutation = FilterTransposePermutation(kernel_shape.rank());
  auto inv_transpose_permutation =
      xla::InversePermutation(transpose_permutation);
  xla::Shape transposed_weight_shape =
      xla::ShapeUtil::PermuteDimensions(transpose_permutation, kernel_shape);
  xla::XlaOp conv = ConsumeValue(MakeXlaBackpropFilterConvOp(
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
  std::vector<int64_t> conv_dims(broadcast_sizes.size());
  std::iota(conv_dims.begin(), conv_dims.end(), 0);
  // Remove the channels dimension.
  broadcast_sizes.erase(broadcast_sizes.begin() + 1);
  conv_dims.erase(conv_dims.begin() + 1);
  // Make the bias match the output dimensions.
  const xla::Shape& bias_shape = ShapeHelper::ShapeOfXlaOp(bias);
  bool bias_broadcast_unbounded_dynamic =
      std::any_of(
          broadcast_sizes.begin(), broadcast_sizes.end(),
          [](int64_t size) { return size == xla::Shape::kUnboundedSize; }) ||
      bias_shape.is_unbounded_dynamic();

  xla::XlaOp broadcasted_bias =
      bias_broadcast_unbounded_dynamic
          ? XlaHelpers::DynamicUnboundedBroadcast(bias, conv, conv_dims)
          : xla::Broadcast(bias, broadcast_sizes);
  xla::XlaOp bias_broadcast = xla::Transpose(
      broadcasted_bias, BiasTransposePermutation(broadcast_sizes.size() + 1));
  const xla::Shape& conv_shape = ShapeHelper::ShapeOfXlaOp(conv);
  const xla::Shape& bb_shape = ShapeHelper::ShapeOfXlaOp(bias_broadcast);
  std::cout << "check conv_shape " << conv_shape << std::endl;
  std::cout << "check bias shape " << bias_shape << std::endl;
  std::cout << "check bb_shape " << bb_shape << std::endl;
  auto promoted = XlaHelpers::Promote(conv, bias_broadcast);
  return xla::Add(
      promoted.first, promoted.second,
      XlaHelpers::getBroadcastDimensions(promoted.first, promoted.second));
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
