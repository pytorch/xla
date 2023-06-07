#include "torch_xla/csrc/convolution.h"

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/client/lib/constants.h"
#include "xla/client/lib/arithmetic.h"
#include "xla/client/xla_builder.h"
#include "xla/literal_util.h"
#include "xla/stream_executor/dnn.h"
#include "third_party/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/xla_lower_util.h"


namespace torch_xla {
namespace {
// Converts the tensor data format to the one required by the XLA convolution
// library.
xla::ConvolutionDimensionNumbers MakeConvolutionDimensionNumbers(
    torch_xla::XLATensorFormat data_format, int num_spatial_dims) {
  int num_dims = num_spatial_dims + 2;
  int batch_dimension = GetTensorBatchDimIndex(num_dims, data_format);
  int feature_dimension = GetTensorFeatureDimIndex(num_dims, data_format);
  xla::ConvolutionDimensionNumbers conv_dim_numbers;
  for (int spatial_dim = 0; spatial_dim < num_spatial_dims; ++spatial_dim) {
    conv_dim_numbers.add_input_spatial_dimensions(
        GetTensorSpatialDimIndex(num_dims, data_format, spatial_dim));
  }
  conv_dim_numbers.set_input_batch_dimension(batch_dimension);
  conv_dim_numbers.set_input_feature_dimension(feature_dimension);
  return conv_dim_numbers;
}

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
ConvOpAttrs MakeConvOpAttrs(
    absl::Span<const int64_t> spatial_stride,
    absl::Span<const int64_t> spatial_padding,
    absl::Span<const int64_t> spatial_dilation, bool depthwise) {
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
  conv_op_attrs.padding = ThreePadding::EXPLICIT;
  // https://github.com/tensorflow/tensorflow/blob/ec81825aaf7e848d9f8ddffdf1e0d20aebe9172c/tensorflow/core/util/padding.cc#L40
  // explicit_padding requires to have (spatial_dims + 2) * 2 elements
  conv_op_attrs.explicit_paddings.resize(4);
  for (int spatial_dim = 0; spatial_dim < num_spatial_dims; ++spatial_dim) {
    conv_op_attrs.explicit_paddings.push_back(spatial_padding[spatial_dim]);
    conv_op_attrs.explicit_paddings.push_back(spatial_padding[spatial_dim]);
  }
  conv_op_attrs.data_format = MakeConvolutionDimensionNumbers(
      torch_xla::XLATensorFormat::FORMAT_NCHW, num_spatial_dims);
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
  return ConsumeValue(MakeXlaBackpropInputConvOp(
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


std::string XLAToString(XLATensorFormat format) {
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

namespace {

// string ToString(FilterTensorFormat format) {
//   switch (format) {
//     case FORMAT_HWIO:
//       return "HWIO";
//     case FORMAT_OIHW:
//       return "OIHW";
//     case FORMAT_OHWI:
//       return "OHWI";
//     case FORMAT_OIHW_VECT_I:
//       return "OIHW_VECT_I";
//     default:
//       LOG(FATAL) << "Invalid Filter Format: " << static_cast<int32>(format);
//       return "INVALID_FORMAT";
//   }
// }

// Returns the expanded size of a filter used for depthwise convolution.
// If `shape` is [H, W, ..., M, N] returns [H, W, ..., M, M*N].
xla::Shape ExpandedFilterShapeForDepthwiseConvolution(const xla::Shape& shape) {
  int num_dims = shape.dimensions_size();
  CHECK_GE(num_dims, 2);  // Crash OK
  xla::Shape expanded_shape = shape;
  expanded_shape.set_dimensions(
      num_dims - 1,
      shape.dimensions(num_dims - 2) * shape.dimensions(num_dims - 1));
  return expanded_shape;
}

// Returns the transposed filter for use in BackpropInput of group convolution.
xla::XlaOp TransposeFilterForGroupConvolutionBackpropInput(const xla::XlaOp& filter,
                                                           const xla::Shape& filter_shape,
                                                           tsl::int64 num_groups,
                                                           int num_spatial_dims) {
  // 1. Reshape from [H, W, ..., filter_in_depth, out_depth] to [H, W, ...,
  // filter_in_depth, G, out_depth / G]
  int num_dims = filter_shape.dimensions_size();
  CHECK_GE(num_dims, 2);  // Crash OK
  xla::Shape new_shape = filter_shape;
  new_shape.set_dimensions(num_dims - 1, num_groups);
  new_shape.add_dimensions(filter_shape.dimensions(num_dims - 1) / num_groups);
  xla::XlaOp result = Reshape(filter, new_shape.dimensions());

  // 2. Transpose to [H, W, ..., G, filter_in_depth, out_depth / G]
  std::vector<tsl::int64> transpose_dims(num_dims + 1);
  std::iota(transpose_dims.begin(), transpose_dims.end(), 0);
  std::swap(transpose_dims[num_spatial_dims],
            transpose_dims[num_spatial_dims + 1]);
  result = Transpose(result, transpose_dims);

  // 3. Reshape to [H, W, ..., in_depth, out_depth / G]
  result = Collapse(result, {num_spatial_dims, num_spatial_dims + 1});
  return result;
}

// Returns the transposed input for use in BackpropFilter of group convolution.
// XlaOp TransposeInputForGroupConvolutionBackpropFilter(const XlaOp& input,
//                                                       const Shape& input_shape,
//                                                       tsl::int64 num_groups,
//                                                       int batch_dim,
//                                                       int depth_dim) {
//   // 1. Reshape the depth_dim C into [G, C/G]
//   int num_dims = input_shape.dimensions_size();
//   std::vector<tsl::int64> reshape_dims = input_shape.dimensions();
//   reshape_dims[depth_dim] = reshape_dims[depth_dim] / num_groups;
//   reshape_dims.insert(reshape_dims.begin() + depth_dim, num_groups);
//   XlaOp result = Reshape(input, reshape_dims);

//   // 2. Transpose G to the axis before N, e.g.: [G, N, H, W, C/G]
//   std::vector<tsl::int64> transpose_dims(num_dims + 1);
//   std::iota(transpose_dims.begin(), transpose_dims.end(),
//             0);  // e.g.: [0, 1, 2, 3, 4] -> [N, H, W, G, C/G]
//   transpose_dims.erase(transpose_dims.begin() + depth_dim);
//   transpose_dims.insert(
//       transpose_dims.begin() + batch_dim,
//       depth_dim);  // e.g.: [3, 0, 1, 2, 4] -> [G, N, H, W, C/G]
//   result = Transpose(result, transpose_dims);

//   // 3. Merge [G, N] to [G*N]
//   result = Collapse(result, {batch_dim, batch_dim + 1});
//   return result;
// }

// Create a mask for depthwise convolution that will make a normal convolution
// produce the same results as a depthwise convolution. For a [2, 2, 3, 2]
// depthwise filter this returns a [2, 2, 3, 6] tensor
//   1 1 0 0 0 0   1 1 0 0 0 0
//   0 0 1 1 0 0   0 0 1 1 0 0
//   0 0 0 0 1 1   0 0 0 0 1 1
//
//   1 1 0 0 0 0   1 1 0 0 0 0
//   0 0 1 1 0 0   0 0 1 1 0 0
//   0 0 0 0 1 1   0 0 0 0 1 1
//
// The first step is to create a iota A with iota_dimension = 2
//   0 0 0 0 0 0   0 0 0 0 0 0
//   1 1 1 1 1 1   1 1 1 1 1 1
//   2 2 2 2 2 2   2 2 2 2 2 2
//
//   0 0 0 0 0 0   0 0 0 0 0 0
//   1 1 1 1 1 1   1 1 1 1 1 1
//   2 2 2 2 2 2   2 2 2 2 2 2
//
// and another iota B with iota_dimension = 3
//   0 1 2 3 4 5  0 1 2 3 4 5
//   0 1 2 3 4 5  0 1 2 3 4 5
//   0 1 2 3 4 5  0 1 2 3 4 5
//
//   0 1 2 3 4 5  0 1 2 3 4 5
//   0 1 2 3 4 5  0 1 2 3 4 5
//   0 1 2 3 4 5  0 1 2 3 4 5
//
// and divide B by 2 to get
//   0 0 1 1 2 2  0 0 1 1 2 2
//   0 0 1 1 2 2  0 0 1 1 2 2
//   0 0 1 1 2 2  0 0 1 1 2 2
//
//   0 0 1 1 2 2  0 0 1 1 2 2
//   0 0 1 1 2 2  0 0 1 1 2 2
//   0 0 1 1 2 2  0 0 1 1 2 2
//
// Finally compare A and B and return the result at the beginning of the
// comment.
xla::XlaOp CreateExpandedFilterMask(const xla::Shape& filter_shape, xla::XlaBuilder* builder) {
  xla::Shape expanded_filter_shape =
      ExpandedFilterShapeForDepthwiseConvolution(filter_shape);
  tsl::int64 depthwise_multiplier =
      filter_shape.dimensions(filter_shape.dimensions_size() - 1);

  // Create two iotas with the shape of the expanded filter, one of them with
  // the iota dimension chosen as the feature dimension, and the other a iota
  // with the iota dimension chosen as the expanded output feature dimension.
  std::vector<tsl::int64> iota_dimensions(expanded_filter_shape.dimensions().begin(),
                                     expanded_filter_shape.dimensions().end());
  xla::Shape iota_shape = xla::ShapeUtil::MakeShape(xla::S32, iota_dimensions);
  xla::XlaOp input_feature_iota =
      Iota(builder, iota_shape, /*iota_dimension=*/iota_dimensions.size() - 2);
  xla::XlaOp expanded_feature_iota =
      Iota(builder, iota_shape, /*iota_dimension=*/iota_dimensions.size() - 1);

  // Divide 'expanded_feature_iota' by the depthwise_multiplier to create
  // [0 0 1 1 2 2] ... in the example in the function comment.
  expanded_feature_iota = Div(
      expanded_feature_iota,
      ConstantR0WithType(builder, xla::PrimitiveType::S32, depthwise_multiplier));

  // Compare 'input_feature_iota' with 'expanded_feature_iota' to create a
  // diagonal predicate.
  return Eq(expanded_feature_iota, input_feature_iota);
}

// Reshapes a filter of shape [H, W, ..., M, N] to [H, W, ..., 1, M*N]. Used to
// build a depthwise convolution.
xla::XlaOp ReshapeFilterForDepthwiseConvolution(const xla::Shape& filter_shape,
                                                const xla::XlaOp& filter) {
  tsl::int64 input_feature_dim = filter_shape.dimensions_size() - 2;
  tsl::int64 output_feature_dim = filter_shape.dimensions_size() - 1;
  tsl::int64 depthwise_multiplier = filter_shape.dimensions(output_feature_dim);
  tsl::int64 input_feature = filter_shape.dimensions(input_feature_dim);

  // Create a [H, W, ..., 1, N*M] reshape of the filter.
  xla::Shape implicit_broadcast_filter_shape = filter_shape;
  implicit_broadcast_filter_shape.set_dimensions(input_feature_dim, 1);
  implicit_broadcast_filter_shape.set_dimensions(
      output_feature_dim, depthwise_multiplier * input_feature);
  return Reshape(filter,
                 stream_executor::dnn::AsInt64Slice(implicit_broadcast_filter_shape.dimensions()));
}

// Reduces the results of the convolution with an expanded filter to the
// non-expanded filter.
xla::XlaOp ContractFilterForDepthwiseBackprop(const xla::Shape& filter_shape,
                                              const xla::XlaOp& filter_backprop,
                                              xla::XlaBuilder* builder) {
  auto masked_expanded_filter =
      Select(CreateExpandedFilterMask(filter_shape, builder), filter_backprop,
             ZerosLike(filter_backprop));

  auto elem_type = filter_shape.element_type();
  return Reshape(
      // This reduce does not need inputs to be converted with
      // XlaHelpers::SumAccumulationType() since the select above guarantees
      // that only one element is non zero, so there cannot be accumulated
      // precision error.
      Reduce(masked_expanded_filter, Zero(builder, elem_type),
             CreateScalarAddComputation(elem_type, builder),
             {filter_shape.dimensions_size() - 2}),
      stream_executor::dnn::AsInt64Slice(filter_shape.dimensions()));
}

// Performs some basic checks on ConvOpAttrs that are true for all kinds of XLA
// convolutions (as currently implemented).
tsl::Status CheckConvAttrs(const ConvOpAttrs& attrs) {
  const int num_dims = attrs.num_spatial_dims + 2;
  if (attrs.strides.size() != num_dims) {
    return tsl::errors::InvalidArgument(
        "Sliding window strides field must specify %d dimensions", num_dims);
  }
  int batch_dim = attrs.data_format.input_batch_dimension();
  int feature_dim = attrs.data_format.input_feature_dimension();
  if (attrs.strides[batch_dim] != 1 || attrs.strides[feature_dim] != 1) {
    return tsl::errors::Unimplemented(
        "Current implementation does not yet support strides in the batch and "
        "depth dimensions.");
  }
  if (attrs.dilations.size() != num_dims) {
    return tsl::errors::InvalidArgument("Dilations field must specify %d dimensions",
                           num_dims);
  }
  if (attrs.dilations[batch_dim] != 1 || attrs.dilations[feature_dim] != 1) {
    return tsl::errors::Unimplemented(
        "Current implementation does not support dilations in the batch and "
        "depth dimensions.");
  }
  for (int i = 0; i < attrs.num_spatial_dims; ++i) {
    int input_dim = attrs.data_format.input_spatial_dimensions(i);
    if (attrs.dilations[input_dim] < 1) {
      return tsl::errors::Unimplemented(
          "Dilation values must be positive; %dth spatial dimension had "
          "dilation %d",
          i, attrs.dilations[input_dim]);
    }
  }
  return tsl::OkStatus();
}

// Information about a single spatial dimension for a convolution
// backpropagation.
struct ConvBackpropSpatialDimension {
  tsl::int64 input_size;
  tsl::int64 filter_size;
  tsl::int64 output_size;
  tsl::int64 stride;
  tsl::int64 dilation;
  // Output size after scaling by the stride.
  tsl::int64 expanded_output_size;
  // Number of padding elements to be added before/after this dimension of
  // the input when computing Conv?DBackpropInput.
  tsl::int64 pad_before, pad_after;
};

// Computed dimensions for a backwards convolution.
struct ConvBackpropDimensions {
  // Information about each spatial dimension.
  std::vector<ConvBackpropSpatialDimension> spatial_dims;
  // Batch size.
  tsl::int64 batch_size;
  // Input and output feature depth.
  tsl::int64 in_depth, out_depth;
};

tsl::Status ConvBackpropExtractAndVerifyDimension(
    absl::Span<const tsl::int64> input_shape, absl::Span<const tsl::int64> filter_shape,
    absl::Span<const tsl::int64> output_shape, absl::Span<const tsl::int32> dilations,
    const std::vector<tsl::int32>& strides, tsl::int64 padding_before,
    tsl::int64 padding_after, int spatial_dim, int filter_spatial_dim,
    ConvBackpropSpatialDimension* dim) {
  dim->input_size = input_shape.at(spatial_dim);
  dim->filter_size = filter_shape.at(filter_spatial_dim);
  dim->output_size = output_shape.at(spatial_dim);
  dim->stride = strides[spatial_dim];
  dim->dilation = dilations[spatial_dim];
  tsl::int64 effective_filter_size = (dim->filter_size - 1) * dim->dilation + 1;
  tsl::int64 out_size = (dim->input_size + padding_before + padding_after -
                    effective_filter_size + dim->stride) /
                   dim->stride;
  if (dim->output_size != out_size) {
    return tsl::errors::InvalidArgument(
        "ConvBackpropExtractAndVerifyDimension: Size of out_backprop doesn't "
        "match computed: actual = %ld, "
        "computed = %ld, spatial_dim: %d, input: %ld, filter: %ld, output: "
        "%ld, stride: %ld, dilation: %ld",
        dim->output_size, out_size, spatial_dim, dim->input_size,
        dim->filter_size, dim->output_size, dim->stride, dim->dilation);
  }

  dim->expanded_output_size = (dim->output_size - 1) * dim->stride + 1;
  const auto padded_out_size = dim->input_size + effective_filter_size - 1;
  dim->pad_before = effective_filter_size - 1 - padding_before;
  dim->pad_after =
      padded_out_size - dim->expanded_output_size - dim->pad_before;
  VLOG(2) << "ConvBackpropExtractAndVerifyDimension: expanded_out = "
          << dim->expanded_output_size
          << ", effective_filter_size = " << effective_filter_size
          << ", padded_out = " << padded_out_size
          << ", pad_before = " << dim->pad_before
          << ", pad_after = " << dim->pad_after
          << ", dilation = " << dim->dilation << ", strides = " << dim->stride;
  return tsl::OkStatus();
}

// Verifies that the dimensions all match, and computes sizes/padding for the
// spatial dimensions.
tsl::Status ConvBackpropComputeDimensions(
    absl::string_view label, int num_spatial_dims,
    absl::Span<const tsl::int64> input_shape, absl::Span<const tsl::int64> filter_shape,
    absl::Span<const tsl::int64> out_backprop_shape,
    absl::Span<const tsl::int32> dilations, const std::vector<tsl::int32>& strides,
    absl::Span<const tsl::int64> explicit_paddings,
    const xla::ConvolutionDimensionNumbers& data_format,
    ConvBackpropDimensions* dims) {
  // The + 2 in the following line is for the batch and feature dimensions.
  const int num_dims = num_spatial_dims + 2;
  if (input_shape.size() != num_dims) {
    return tsl::errors::InvalidArgument("%s: input must be %d-dimensional", label, num_dims);
  }
  if (filter_shape.size() != num_dims) {
    return tsl::errors::InvalidArgument("%s: filter must be %d-dimensional", label,
                           num_dims);
  }
  if (out_backprop_shape.size() != num_dims) {
    return tsl::errors::InvalidArgument("%s: out_backprop must be %d-dimensional", label,
                           num_dims);
  }
  int batch_dim = data_format.input_batch_dimension();
  dims->batch_size = input_shape.at(batch_dim);
  if (dims->batch_size != out_backprop_shape.at(batch_dim)) {
    return tsl::errors::InvalidArgument(
        "%s: input and out_backprop must have the same batch size, input "
        "batch: %ld outbackprop batch: %ld batch_dim: %d",
        label, dims->batch_size, out_backprop_shape.at(batch_dim), batch_dim);
  }

  int feature_dim = data_format.input_feature_dimension();
  dims->in_depth = input_shape.at(feature_dim);
  // The input and output feature dimensions are the second last and last
  // dimensions of the filter Tensor.
  VLOG(2) << "input vs filter_in depth " << dims->in_depth << " "
          << filter_shape.at(num_dims - 2);
  if (dims->in_depth % filter_shape.at(num_dims - 2)) {
    return tsl::errors::InvalidArgument(
        "%s: input depth must be evenly divisible by filter depth", label);
  }
  dims->out_depth = filter_shape.at(num_dims - 1);
  if (dims->out_depth != out_backprop_shape.at(feature_dim)) {
    return tsl::errors::InvalidArgument(
        "%s: filter and out_backprop must have the same out_depth", label);
  }
  dims->spatial_dims.resize(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    int image_dim = data_format.input_spatial_dimensions(i);
    tsl::int64 padding_before = -1, padding_after = -1;
    padding_before = explicit_paddings[2 * image_dim];
    padding_after = explicit_paddings[2 * image_dim + 1];
    TF_RETURN_IF_ERROR(ConvBackpropExtractAndVerifyDimension(
        input_shape, filter_shape, out_backprop_shape, dilations, strides,
        padding_before, padding_after, image_dim, i, &dims->spatial_dims[i]));
  }
  return tsl::OkStatus();
}

}  // anonymous namespace

tsl::StatusOr<xla::XlaOp> MakeXlaForwardConvOp(absl::string_view /*type_string*/,
                                          xla::XlaOp conv_input, xla::XlaOp filter,
                                          const ConvOpAttrs& attrs,
                                          const xla::PrecisionConfig* precision_config) {
  TF_RETURN_IF_ERROR(CheckConvAttrs(attrs));

  auto* builder = conv_input.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape input_shape, builder->GetShape(conv_input));
  // Filter has the form [filter_rows, filter_cols, ..., in_depth, out_depth]
  TF_ASSIGN_OR_RETURN(xla::Shape filter_shape, builder->GetShape(filter));

  // For 2D convolution, there should be 4 dimensions.
  int num_dims = attrs.num_spatial_dims + 2;
  if (input_shape.dimensions_size() != num_dims) {
    return tsl::errors::InvalidArgument("input must be %d-dimensional: %s", num_dims,
                           input_shape.DebugString());
  }
  if (filter_shape.dimensions_size() != num_dims) {
    return tsl::errors::InvalidArgument("filter must be %d-dimensional: %s", num_dims,
                           filter_shape.DebugString());
  }

  // The last two dimensions of the filter are the input and output shapes.
  int batch_dim = attrs.data_format.input_batch_dimension();
  int feature_dim = attrs.data_format.input_feature_dimension();

  tsl::int64 filter_in_depth = filter_shape.dimensions(attrs.num_spatial_dims),
        out_depth = filter_shape.dimensions(attrs.num_spatial_dims + 1),
        in_depth = input_shape.dimensions(feature_dim);
  // The 'C' dimension for input is in_depth.
  // It must be a multiple of the filter's in_depth.
  if (in_depth % filter_in_depth != 0) {
    return tsl::errors::InvalidArgument(
        "Depth of input must be a multiple of depth of filter: %d vs %d",
        in_depth, filter_in_depth);
  }
  tsl::int64 feature_group_count = in_depth / filter_in_depth;
  if (out_depth % feature_group_count != 0) {
    return tsl::errors::InvalidArgument(
        "Depth of output must be a multiple of the number of groups: %d vs %d",
        out_depth, feature_group_count);
  }

  if (attrs.depthwise) {
    filter = ReshapeFilterForDepthwiseConvolution(filter_shape, filter);
  }

  xla::ConvolutionDimensionNumbers dims;
  std::vector<tsl::int64> window_strides(attrs.num_spatial_dims);
  std::vector<tsl::int64> lhs_dilation(attrs.num_spatial_dims, 1);
  std::vector<tsl::int64> rhs_dilation(attrs.num_spatial_dims);
  std::vector<std::pair<tsl::int64, tsl::int64>> padding(attrs.num_spatial_dims);

  dims.set_input_batch_dimension(batch_dim);
  dims.set_output_batch_dimension(batch_dim);
  dims.set_input_feature_dimension(feature_dim);
  dims.set_output_feature_dimension(feature_dim);
  dims.set_kernel_input_feature_dimension(attrs.num_spatial_dims);
  dims.set_kernel_output_feature_dimension(attrs.num_spatial_dims + 1);

  for (int i = 0; i < attrs.num_spatial_dims; ++i) {
    const tsl::int64 dim = attrs.data_format.input_spatial_dimensions(i);
    dims.add_input_spatial_dimensions(dim);
    dims.add_kernel_spatial_dimensions(i);
    dims.add_output_spatial_dimensions(dim);
    window_strides[i] = attrs.strides.at(dim);
    rhs_dilation[i] = attrs.dilations.at(dim);
    padding[i] = {attrs.explicit_paddings.at(dim * 2),
                  attrs.explicit_paddings.at(dim * 2 + 1)};
  }

  return ConvGeneralDilated(
      conv_input, filter, window_strides, padding, lhs_dilation, rhs_dilation,
      dims,
      /*feature_group_count=*/attrs.depthwise ? in_depth : feature_group_count,
      /*batch_group_count=*/1, precision_config);
}

tsl::StatusOr<xla::XlaOp> MakeXlaBackpropInputConvOp(
    absl::string_view type_string, const xla::Shape& input_shape, xla::XlaOp filter,
    xla::XlaOp out_backprop, const ConvOpAttrs& attrs,
    const xla::PrecisionConfig* precision_config) {
  TF_RETURN_IF_ERROR(CheckConvAttrs(attrs));

  int batch_dim = attrs.data_format.input_batch_dimension();
  int feature_dim = attrs.data_format.input_feature_dimension();

  auto* builder = filter.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape filter_shape, builder->GetShape(filter));
  TF_ASSIGN_OR_RETURN(xla::Shape out_backprop_shape,
                      builder->GetShape(out_backprop));

  tsl::int64 in_depth = input_shape.dimensions(feature_dim),
        filter_in_depth = filter_shape.dimensions(attrs.num_spatial_dims),
        feature_group_count = in_depth / filter_in_depth;

  xla::Shape expanded_filter_shape =
      attrs.depthwise ? ExpandedFilterShapeForDepthwiseConvolution(filter_shape)
                      : filter_shape;
  // Reuse dimension computation logic from conv_grad_ops.cc.
  ConvBackpropDimensions dims;
  TF_RETURN_IF_ERROR(ConvBackpropComputeDimensions(
      type_string, attrs.num_spatial_dims, input_shape.dimensions(),
      expanded_filter_shape.dimensions(), out_backprop_shape.dimensions(),
      attrs.dilations, attrs.strides, attrs.explicit_paddings,
      attrs.data_format, &dims));

  // The input gradients are computed by a convolution of the output
  // gradients and the filter, with some appropriate padding. See the
  // comment at the top of conv_grad_ops.h for details.

  xla::ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(batch_dim);
  dnums.set_output_batch_dimension(batch_dim);
  dnums.set_input_feature_dimension(feature_dim);
  dnums.set_output_feature_dimension(feature_dim);

  // TF filter shape is [ H, W, ..., inC, outC ]
  // Transpose the input and output features for computing the gradient.
  dnums.set_kernel_input_feature_dimension(attrs.num_spatial_dims + 1);
  dnums.set_kernel_output_feature_dimension(attrs.num_spatial_dims);

  std::vector<tsl::int64> kernel_spatial_dims(attrs.num_spatial_dims);
  std::vector<std::pair<tsl::int64, tsl::int64>> padding(attrs.num_spatial_dims);
  std::vector<tsl::int64> lhs_dilation(attrs.num_spatial_dims);
  std::vector<tsl::int64> rhs_dilation(attrs.num_spatial_dims);
  std::vector<tsl::int64> ones(attrs.num_spatial_dims, 1);
  for (int i = 0; i < attrs.num_spatial_dims; ++i) {
    tsl::int64 dim = attrs.data_format.input_spatial_dimensions(i);
    dnums.add_input_spatial_dimensions(dim);
    dnums.add_kernel_spatial_dimensions(i);
    dnums.add_output_spatial_dimensions(dim);

    kernel_spatial_dims[i] = i;
    padding[i] = {dims.spatial_dims[i].pad_before,
                  dims.spatial_dims[i].pad_after};
    lhs_dilation[i] = dims.spatial_dims[i].stride;
    rhs_dilation[i] = attrs.dilations[dim];
  }

  if (feature_group_count != 1 && !attrs.depthwise) {
    filter = TransposeFilterForGroupConvolutionBackpropInput(
        filter, filter_shape, feature_group_count, attrs.num_spatial_dims);
  }
  // Mirror the filter in the spatial dimensions.
  filter = Rev(filter, kernel_spatial_dims);

  // activation gradients
  //   = gradients (with padding and dilation) <conv> mirrored_weights
  return ConvGeneralDilated(
      out_backprop, filter, /*window_strides=*/ones, padding, lhs_dilation,
      rhs_dilation, dnums,
      /*feature_group_count=*/
      attrs.depthwise ? out_backprop_shape.dimensions(feature_dim) /
                            filter_shape.dimensions(attrs.num_spatial_dims + 1)
                      : feature_group_count,
      /*batch_group_count=*/1, precision_config);
}

tsl::StatusOr<xla::XlaOp> MakeXlaBackpropFilterConvOp(
    absl::string_view type_string, xla::XlaOp activations, const xla::Shape& filter_shape,
    xla::XlaOp out_backprop, const ConvOpAttrs& attrs,
    const xla::PrecisionConfig* precision_config) {
  TF_RETURN_IF_ERROR(CheckConvAttrs(attrs));

  auto* builder = activations.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape activations_shape, builder->GetShape(activations));
  TF_ASSIGN_OR_RETURN(xla::Shape out_backprop_shape,
                      builder->GetShape(out_backprop));
  xla::XlaOp filter_backprop;

  xla::Shape input_shape = activations_shape;
  xla::Shape output_shape = out_backprop_shape;

  const xla::Shape expanded_filter_shape =
      attrs.depthwise ? ExpandedFilterShapeForDepthwiseConvolution(filter_shape)
                      : filter_shape;
  // Reuse dimension computation logic from conv_grad_ops.cc.
  ConvBackpropDimensions dims;
  // The filter gradients are computed by a convolution of the input
  // activations and the output gradients, with some appropriate padding.
  // See the comment at the top of conv_grad_ops.h for details.
  xla::ConvolutionDimensionNumbers dnums;

  TF_RETURN_IF_ERROR(ConvBackpropComputeDimensions(
      type_string, attrs.num_spatial_dims, activations_shape.dimensions(),
      expanded_filter_shape.dimensions(), out_backprop_shape.dimensions(),
      attrs.dilations, attrs.strides, attrs.explicit_paddings,
      attrs.data_format, &dims));

  // Obtain some useful dimensions:
  // The last two dimensions of the filter are the input and output shapes.
  int num_dims = attrs.num_spatial_dims + 2;
  int n_dim = attrs.data_format.input_batch_dimension();
  int c_dim = attrs.data_format.input_feature_dimension();
  tsl::int64 in_depth = input_shape.dimensions(c_dim),
        filter_in_depth = filter_shape.dimensions(attrs.num_spatial_dims),
        feature_group_count = in_depth / filter_in_depth;

  // The activations (inputs) form the LHS of the convolution.
  // Activations have shape: [batch, in_rows, in_cols, ..., in_depth]
  // For the gradient computation, we need to:
  // 1. In the case of group convolution, move the num_groups dimension before
  // the batch dimension
  // 2. Swap the roles of the batch and feature dimensions.
//   if (feature_group_count != 1 && !attrs.depthwise) {
//     activations = TransposeInputForGroupConvolutionBackpropFilter(
//         activations, input_shape, feature_group_count, n_dim, c_dim);
//   }

  // In the case of depthwise convolution with no multiplier,
  // the computation can be done by the batch_group_count parameter.
  bool use_batch_group_count =
      filter_shape.dimensions(num_dims - 1) == 1 && attrs.depthwise;

  std::vector<std::pair<tsl::int64, tsl::int64>> padding(attrs.num_spatial_dims);
  std::vector<tsl::int64> rhs_dilation(attrs.num_spatial_dims);
  std::vector<tsl::int64> window_strides(attrs.num_spatial_dims);
  std::vector<tsl::int64> ones(attrs.num_spatial_dims, 1);

  // Swap n_dim and c_dim in the activations.
  dnums.set_input_batch_dimension(c_dim);
  dnums.set_input_feature_dimension(n_dim);

  // The gradients become the RHS of the convolution.
  // The gradients have shape [batch, out_rows, out_cols, ..., out_depth]
  // where the batch becomes the input feature for the convolution.
  dnums.set_kernel_input_feature_dimension(n_dim);
  dnums.set_kernel_output_feature_dimension(c_dim);

  // The dimension swap below is needed because filter shape is KH,KW,F,DM.
  if (use_batch_group_count) {
    dnums.set_output_batch_dimension(attrs.num_spatial_dims + 1);
    dnums.set_output_feature_dimension(attrs.num_spatial_dims);
  } else {
    dnums.set_output_batch_dimension(attrs.num_spatial_dims);
    dnums.set_output_feature_dimension(attrs.num_spatial_dims + 1);
  }

  // Tensorflow filter shape is [ H, W, ..., inC, outC ].
  for (int i = 0; i < attrs.num_spatial_dims; ++i) {
    dnums.add_output_spatial_dimensions(i);
  }

  for (tsl::int64 i = 0; i < attrs.num_spatial_dims; ++i) {
    tsl::int64 dim = attrs.data_format.input_spatial_dimensions(i);
    dnums.add_input_spatial_dimensions(dim);
    dnums.add_kernel_spatial_dimensions(dim);
    rhs_dilation[i] = dims.spatial_dims[i].stride;
    window_strides[i] = attrs.dilations[dim];

    // We will also need to pad the input with zeros such that after the
    // convolution, we get the right size for the filter.
    // The padded_in_rows should be such that when we convolve this with the
    // expanded_out_rows as a filter, we should get filter_rows back.

    const tsl::int64 padded_in_size =
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
    const tsl::int64 pad_total = padded_in_size - dims.spatial_dims[i].input_size;

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
    const tsl::int64 pad_before = attrs.explicit_paddings[2 * dim];
    padding[i] = {pad_before, pad_total - pad_before};
  }

  // Besides padding the input, we will also expand output_rows to
  //    expanded_out_rows = (output_rows - 1) * stride + 1
  // with zeros in between:
  //
  //      a . . . b . . . c . . . d . . . e
  //
  // This is done by specifying the window dilation factors in the
  // convolution HLO below.

  filter_backprop = ConvGeneralDilated(
      activations, out_backprop, window_strides, padding, /*lhs_dilation=*/ones,
      rhs_dilation, dnums,
      /*feature_group_count=*/feature_group_count,
      /*batch_group_count=*/use_batch_group_count ? dims.in_depth : 1,
      precision_config);

  if (!use_batch_group_count && attrs.depthwise) {
    filter_backprop = ContractFilterForDepthwiseBackprop(
        filter_shape, filter_backprop, activations.builder());
  }

  return filter_backprop;
}

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
