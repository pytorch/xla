#include "torch_xla/csrc/convolution.h"

#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "third_party/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {
namespace {

// Create a TF convolution metadata structure out of PyTorch convolution
// attributes.
tensorflow::ConvOpAttrs MakeConvOpAttrs(
    tensorflow::gtl::ArraySlice<const xla::int64> spatial_stride,
    tensorflow::gtl::ArraySlice<const xla::int64> spatial_padding,
    tensorflow::gtl::ArraySlice<const xla::int64> spatial_dilation,
    bool depthwise) {
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
const std::vector<xla::int64>& FilterTransposePermutation(const xla::int64 k) {
  if (k == 4) {
    static std::vector<xla::int64>* permutation =
        new std::vector<xla::int64>({2, 3, 1, 0});
    return *permutation;
  } else if (k == 5) {
    static std::vector<xla::int64>* permutation =
        new std::vector<xla::int64>({2, 3, 4, 1, 0});
    return *permutation;
  } else {
    XLA_ERROR() << "Invalid rank: " << k;
  }
}

// Bias broadcast based on output shape produces:
// (N, H, W) + (C,) = (N, H, W, C)
// This permutation does (N, H, W, C) -> (N, C, H, W)
const std::vector<xla::int64>& BiasTransposePermutation(const xla::int64 k) {
  if (k == 4) {
    static std::vector<xla::int64>* permutation =
        new std::vector<xla::int64>({0, 3, 1, 2});
    return *permutation;
  } else if (k == 5) {
    static std::vector<xla::int64>* permutation =
        new std::vector<xla::int64>({0, 4, 1, 2, 3});
    return *permutation;
  } else {
    XLA_ERROR() << "Invalid rank: " << k;
  }
}

// Reduce bias from (N, C, H, W) to (C,)
const std::vector<xla::int64>& BiasReduceDimensions(const xla::int64 k) {
  if (k == 4) {
    static std::vector<xla::int64>* reduce_dim =
        new std::vector<xla::int64>({0, 2, 3});
    return *reduce_dim;
  } else if (k == 5) {
    static std::vector<xla::int64>* reduce_dim =
        new std::vector<xla::int64>({0, 2, 3, 4});
    return *reduce_dim;
  } else {
    XLA_ERROR() << "Invalid rank: " << k;
  }
}

// Computes the input gradient for a convolution.
xla::XlaOp BuildConvBackwardInput(
    const xla::XlaOp& grad_output, const xla::XlaOp& kernel,
    const xla::Shape& input_shape,
    tensorflow::gtl::ArraySlice<const xla::int64> spatial_stride,
    tensorflow::gtl::ArraySlice<const xla::int64> spatial_padding,
    tensorflow::gtl::ArraySlice<const xla::int64> spatial_dilation,
    xla::int64 groups) {
  bool depthwise = groups == input_shape.dimensions(1);
  tensorflow::ConvOpAttrs conv_op_attrs = MakeConvOpAttrs(
      spatial_stride, spatial_padding, spatial_dilation, depthwise);
  xla::XlaOp kernel_transposed =
      xla::Transpose(kernel, FilterTransposePermutation(input_shape.rank()));
  xla::PrecisionConfig precision_config =
      XlaHelpers::BuildPrecisionConfig(XlaHelpers::mat_mul_precision());
  return ConsumeValue(tensorflow::MakeXlaBackpropInputConvOp(
      "conv_backward", input_shape, kernel_transposed, grad_output,
      conv_op_attrs, &precision_config));
}

// Computes the kernel gradient for a convolution.
xla::XlaOp BuildConvBackwardWeight(
    const xla::XlaOp& grad_output, const xla::XlaOp& input,
    const xla::Shape& kernel_shape,
    tensorflow::gtl::ArraySlice<const xla::int64> spatial_stride,
    tensorflow::gtl::ArraySlice<const xla::int64> spatial_padding,
    tensorflow::gtl::ArraySlice<const xla::int64> spatial_dilation,
    xla::int64 groups) {
  bool depthwise = groups == XlaHelpers::ShapeOfXlaOp(input).dimensions(1);
  tensorflow::ConvOpAttrs conv_op_attrs = MakeConvOpAttrs(
      spatial_stride, spatial_padding, spatial_dilation, depthwise);
  auto inv_transpose_permutation =
      xla::InversePermutation(FilterTransposePermutation(kernel_shape.rank()));
  xla::Shape transposed_weight_shape = xla::ShapeUtil::PermuteDimensions(
      inv_transpose_permutation, kernel_shape);
  xla::PrecisionConfig precision_config =
      XlaHelpers::BuildPrecisionConfig(XlaHelpers::mat_mul_precision());
  xla::XlaOp conv = ConsumeValue(tensorflow::MakeXlaBackpropFilterConvOp(
      "conv2d_backward", input, transposed_weight_shape, grad_output,
      conv_op_attrs, &precision_config));

  // Reorder the dimensions of the filter gradient to match the NCHW convention
  // of PyTorch. The original result of the convolution has the spatial and
  // feature dimensions swapped and the spatial dimensions reversed.
  return xla::Transpose(conv, inv_transpose_permutation);
}

xla::XlaOp BuildGradBias(xla::XlaOp grad_output) {
  xla::Shape grad_output_shape = XlaHelpers::ShapeOfXlaOp(grad_output);
  // The bias contribution is linear in each output feature. Reduce the
  // remaining dimensions to get a tensor of the same shape as the bias, rank-1
  // with number of output features elements.
  return xla::Reduce(
      grad_output,
      XlaHelpers::ScalarValue<float>(0, grad_output_shape.element_type(),
                                     grad_output.builder()),
      XlaHelpers::CreateAddComputation(grad_output_shape.element_type()),
      BiasReduceDimensions(grad_output_shape.rank()));
}

}  // namespace

xla::XlaOp BuildTransposedConvolution(
    const xla::XlaOp& input, const xla::XlaOp& kernel,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    tensorflow::gtl::ArraySlice<const xla::int64> dilation,
    tensorflow::gtl::ArraySlice<const xla::int64> output_padding,
    xla::int64 groups) {
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::Shape kernel_shape = XlaHelpers::ShapeOfXlaOp(kernel);
  xla::int64 num_spatial = input_shape.rank() - 2;
  // We only support 2D or 3D convolution.
  XLA_CHECK(num_spatial == 2 || num_spatial == 3) << num_spatial;
  // Fold group into input_size feature dimension
  xla::int64 feature_dim = kernel_shape.dimensions(1) * groups;
  std::vector<xla::int64> input_size{input_shape.dimensions(0), feature_dim};
  for (int spatial_dim = 0; spatial_dim < num_spatial; ++spatial_dim) {
    input_size.push_back(
        (input_shape.dimensions(2 + spatial_dim) - 1) * stride[spatial_dim] -
        2 * padding[spatial_dim] +
        dilation[spatial_dim] * (kernel_shape.dimensions(2 + spatial_dim) - 1) +
        output_padding[spatial_dim] + 1);
  }
  return BuildConvBackwardInput(
      input, kernel,
      xla::ShapeUtil::MakeShape(input_shape.element_type(), input_size), stride,
      padding, dilation, /*groups=*/1);
}

xla::XlaOp BuildTransposedConvolutionBias(
    const xla::XlaOp& input, const xla::XlaOp& kernel, const xla::XlaOp& bias,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    tensorflow::gtl::ArraySlice<const xla::int64> dilation,
    tensorflow::gtl::ArraySlice<const xla::int64> output_padding,
    xla::int64 groups) {
  xla::XlaOp conv = BuildTransposedConvolution(
      input, kernel, stride, padding, dilation, output_padding, groups);
  auto broadcast_sizes = XlaHelpers::SizesOfXlaOp(conv);
  // Remove the channels dimension.
  broadcast_sizes.erase(broadcast_sizes.begin() + 1);
  // Make the bias match the output dimensions.
  xla::XlaOp bias_broadcast =
      xla::Transpose(xla::Broadcast(bias, broadcast_sizes),
                     BiasTransposePermutation(broadcast_sizes.size() + 1));
  return conv + bias_broadcast;
}

ConvGrads BuildTransposedConvolutionBackward(
    const xla::XlaOp& grad_output, const xla::XlaOp& input,
    const xla::XlaOp& kernel,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    tensorflow::gtl::ArraySlice<const xla::int64> dilation,
    tensorflow::gtl::ArraySlice<const xla::int64> output_padding,
    xla::int64 groups) {
  xla::XlaOp grad_input =
      BuildConvolutionOverrideable(grad_output, kernel, stride, padding,
                                   dilation, false, output_padding, groups);
  xla::XlaOp grad_weight = BuildConvBackwardWeight(
      input, grad_output, XlaHelpers::ShapeOfXlaOp(kernel), stride, padding,
      dilation, groups);
  xla::XlaOp grad_bias = BuildGradBias(grad_output);
  return {grad_input, grad_weight, grad_bias};
}

xla::XlaOp BuildConvolutionOverrideable(
    const xla::XlaOp& input, const xla::XlaOp& kernel,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    tensorflow::gtl::ArraySlice<const xla::int64> dilation, bool transposed,
    tensorflow::gtl::ArraySlice<const xla::int64> output_padding,
    xla::int64 groups) {
  if (transposed) {
    return BuildTransposedConvolution(input, kernel, stride, padding, dilation,
                                      output_padding, groups);
  } else {
    xla::PrecisionConfig precision_config =
        XlaHelpers::BuildPrecisionConfig(XlaHelpers::mat_mul_precision());

    xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
    xla::Shape kernel_shape = XlaHelpers::ShapeOfXlaOp(kernel);
    auto Cin = input_shape.dimensions(1);
    auto Cout = kernel_shape.dimensions(0);
    bool depthwise = groups == Cin && Cout % Cin == 0;
    tensorflow::ConvOpAttrs conv_op_attrs = MakeConvOpAttrs(
        stride, padding, dilation, depthwise);
      xla::XlaOp kernel_transposed =
        xla::Transpose(kernel, FilterTransposePermutation(input_shape.rank()));
    xla::XlaOp depth_kernel;
    if (depthwise) {
      std::cout << "DEBUG " << std::endl;
      // FIXME: 3D
      depth_kernel = xla::Reshape(kernel_transposed, {kernel_shape.dimensions(2), kernel_shape.dimensions(3), Cin, Cout/Cin});
    }
    return ConsumeValue(tensorflow::MakeXlaForwardConvOp(
        "conv_forward", input, depthwise ? depth_kernel : kernel_transposed,
        conv_op_attrs, &precision_config));
  }
}

xla::XlaOp BuildConvolutionOverrideableBias(
    const xla::XlaOp& input, const xla::XlaOp& kernel, const xla::XlaOp& bias,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    tensorflow::gtl::ArraySlice<const xla::int64> dilation, bool transposed,
    tensorflow::gtl::ArraySlice<const xla::int64> output_padding,
    xla::int64 groups) {
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
    const xla::XlaOp& grad_output, const xla::XlaOp& input,
    const xla::XlaOp& kernel,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    tensorflow::gtl::ArraySlice<const xla::int64> dilation, bool transposed,
    tensorflow::gtl::ArraySlice<const xla::int64> output_padding,
    xla::int64 groups) {
  if (transposed) {
    return BuildTransposedConvolutionBackward(grad_output, input, kernel,
                                              stride, padding, dilation,
                                              output_padding, groups);
  } else {
    xla::XlaOp grad_input = BuildConvBackwardInput(
        grad_output, kernel, XlaHelpers::ShapeOfXlaOp(input), stride, padding,
        dilation, groups);
    xla::XlaOp grad_weight = BuildConvBackwardWeight(
        grad_output, input, XlaHelpers::ShapeOfXlaOp(kernel), stride, padding,
        dilation, groups);
    xla::XlaOp grad_bias = BuildGradBias(grad_output);
    return {grad_input, grad_weight, grad_bias};
  }
}
}  // namespace torch_xla
