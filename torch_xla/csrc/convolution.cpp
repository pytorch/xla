#include "torch_xla/csrc/convolution.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/translator.h"

namespace torch_xla {
namespace {

// Computes the input gradient for a convolution.
xla::XlaOp BuildThnnConv2dBackwardInput(
    const xla::XlaOp& grad_output, const xla::XlaOp& input,
    const xla::XlaOp& weight,
    tensorflow::gtl::ArraySlice<const xla::int64> stride_attr,
    tensorflow::gtl::ArraySlice<const xla::int64> padding_attr,
    const xla::PrecisionConfig::Precision conv_precision) {
  XLA_CHECK_EQ(stride_attr.size(), 2);
  // Adjust input size to account for specified padding.
  auto input_size = XlaHelpers::SizesOfXlaOp(input);
  for (int i = 0; i < 2; ++i) {
    input_size[2 + i] += 2 * padding_attr[i];
  }
  tensorflow::TensorShape input_shape(input_size);
  xla::XlaOp filter = xla::Transpose(weight, {2, 3, 1, 0});
  xla::XlaBuilder* builder = grad_output.builder();
  const auto filter_size = XlaHelpers::SizesOfXlaOp(filter);
  tensorflow::TensorShape filter_shape(filter_size);
  tensorflow::TensorShape out_backprop_shape(
      XlaHelpers::SizesOfXlaOp(grad_output));
  std::vector<int> strides{1, 1};
  std::copy(stride_attr.begin(), stride_attr.end(),
            std::back_inserter(strides));
  tensorflow::ConvBackpropDimensions dims;
  constexpr int num_spatial_dims = 2;
  std::vector<int> dilations{1, 1, 1, 1};
  xla::Status status = ConvBackpropComputeDimensionsV2(
      "thnn_conv2d_backward", num_spatial_dims, input_shape, filter_shape,
      out_backprop_shape, dilations, strides, tensorflow::Padding::VALID,
      /*explicit_paddings=*/{}, tensorflow::TensorFormat::FORMAT_NCHW, &dims);
  XLA_CHECK_OK(status);

  constexpr int batch_dim = 0;
  constexpr int feature_dim = 1;

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
  dnums.set_kernel_input_feature_dimension(num_spatial_dims + 1);
  dnums.set_kernel_output_feature_dimension(num_spatial_dims);

  std::vector<xla::int64> kernel_spatial_dims(num_spatial_dims);
  std::vector<std::pair<xla::int64, xla::int64>> padding(num_spatial_dims);
  std::vector<xla::int64> lhs_dilation(num_spatial_dims);
  std::vector<xla::int64> rhs_dilation(num_spatial_dims);
  std::vector<xla::int64> ones(num_spatial_dims, 1);
  for (int i = 0; i < num_spatial_dims; ++i) {
    xla::int64 dim = 2 + i;
    dnums.add_input_spatial_dimensions(dim);
    dnums.add_kernel_spatial_dimensions(i);
    dnums.add_output_spatial_dimensions(dim);

    kernel_spatial_dims[i] = i;
    padding[i] = {dims.spatial_dims[i].pad_before,
                  dims.spatial_dims[i].pad_after};
    lhs_dilation[i] = dims.spatial_dims[i].stride;
    rhs_dilation[i] = dilations[dim];
  }

  // Mirror the filter in the spatial dimensions.
  xla::XlaOp mirrored_weights = xla::Rev(filter, kernel_spatial_dims);

  // We'll need to undo the initial input padding once on the input backprop
  // result since edges are constant and have to be discarded for the gradient.
  xla::PaddingConfig padding_config;
  for (int i = 0; i < 2; ++i) {
    padding_config.add_dimensions();
  }
  for (int i = 0; i < 2; ++i) {
    xla::PaddingConfig::PaddingConfigDimension* dims =
        padding_config.add_dimensions();
    dims->set_edge_padding_low(-padding_attr[i]);
    dims->set_edge_padding_high(-padding_attr[i]);
  }

  // activation gradients
  //   = gradients (with padding and dilation) <conv> mirrored_weights
  xla::PrecisionConfig precision_config =
      XlaHelpers::BuildPrecisionConfig(conv_precision);
  xla::Shape weight_shape = XlaHelpers::ShapeOfXlaOp(weight);
  return xla::Pad(
      xla::ConvGeneralDilated(grad_output, mirrored_weights,
                              /*window_strides=*/ones, padding, lhs_dilation,
                              rhs_dilation, dnums,
                              /*feature_group_count=*/1,
                              /*batch_group_count=*/1, &precision_config),
      XlaHelpers::ScalarValue<float>(0, weight_shape.element_type(), builder),
      padding_config);
}

// Computes the weight gradient for a convolution.
xla::XlaOp BuildThnnConv2dBackwardWeight(
    const xla::XlaOp& grad_output, const xla::XlaOp& input,
    const xla::XlaOp& weight,
    tensorflow::gtl::ArraySlice<const xla::int64> stride_attr,
    tensorflow::gtl::ArraySlice<const xla::int64> padding_attr,
    const xla::PrecisionConfig::Precision conv_precision) {
  constexpr int n_dim = 0;
  constexpr int c_dim = 1;
  XLA_CHECK_EQ(padding_attr.size(), 2);
  // Adjust input size to account for specified padding.
  auto input_size = XlaHelpers::SizesOfXlaOp(input);
  for (int i = 0; i < 2; ++i) {
    input_size[2 + i] += 2 * padding_attr[i];
  }
  tensorflow::TensorShape activations_shape(input_size);
  const auto filter_size = XlaHelpers::SizesOfXlaOp(weight);
  std::vector<xla::int64> filter_size_backward{filter_size[2], filter_size[3],
                                               filter_size[1], filter_size[0]};
  tensorflow::TensorShape filter_shape(filter_size_backward);
  tensorflow::TensorShape out_backprop_shape(
      XlaHelpers::SizesOfXlaOp(grad_output));
  std::vector<int> strides{1, 1};
  std::copy(stride_attr.begin(), stride_attr.end(),
            std::back_inserter(strides));
  tensorflow::ConvBackpropDimensions dims;
  constexpr int num_spatial_dims = 2;
  std::vector<int> dilations{1, 1, 1, 1};
  xla::Status status = ConvBackpropComputeDimensionsV2(
      "thnn_conv2d_backward", num_spatial_dims, activations_shape, filter_shape,
      out_backprop_shape, dilations, strides, tensorflow::Padding::VALID,
      /*explicit_paddings=*/{}, tensorflow::TensorFormat::FORMAT_NCHW, &dims);
  XLA_CHECK(status.ok()) << status.error_message();

  // The filter gradients are computed by a convolution of the input
  // activations and the output gradients, with some appropriate padding.
  // See the comment at the top of conv_grad_ops.h for details.

  xla::ConvolutionDimensionNumbers dnums;

  // The activations (inputs) form the LHS of the convolution.
  // Activations have shape: [batch, in_rows, in_cols, ..., in_depth]
  // For the gradient computation, we flip the roles of the batch and
  // feature dimensions.
  // Each spatial entry has size in_depth * batch

  // Swap n_dim and c_dim in the activations.
  dnums.set_input_batch_dimension(c_dim);
  dnums.set_input_feature_dimension(n_dim);

  // The gradients become the RHS of the convolution.
  // The gradients have shape [batch, out_rows, out_cols, ..., out_depth]
  // where the batch becomes the input feature for the convolution.
  dnums.set_kernel_input_feature_dimension(n_dim);
  dnums.set_kernel_output_feature_dimension(c_dim);

  std::vector<std::pair<xla::int64, xla::int64>> padding(num_spatial_dims);
  std::vector<xla::int64> rhs_dilation(num_spatial_dims);
  std::vector<xla::int64> window_strides(num_spatial_dims);
  std::vector<xla::int64> ones(num_spatial_dims, 1);

  // Tensorflow filter shape is [ H, W, ..., inC, outC ].
  for (int i = 0; i < num_spatial_dims; ++i) {
    dnums.add_output_spatial_dimensions(i);
  }
  dnums.set_output_batch_dimension(num_spatial_dims);
  dnums.set_output_feature_dimension(num_spatial_dims + 1);

  for (int i = 0; i < num_spatial_dims; ++i) {
    xla::int64 dim = 2 + i;
    dnums.add_input_spatial_dimensions(dim);
    dnums.add_kernel_spatial_dimensions(dim);

    // We will also need to pad the input with zeros such that after the
    // convolution, we get the right size for the filter.
    // The padded_in_rows should be such that when we convolve this with the
    // expanded_out_rows as a filter, we should get filter_rows back.
    //
    const xla::int64 padded_in_size =
        dims.spatial_dims[i].expanded_output_size +
        (dims.spatial_dims[i].filter_size - 1) * dilations[dim];

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
    const xla::int64 pad_total =
        padded_in_size - dims.spatial_dims[i].input_size;

    // Pad the bottom/right side with the remaining space.
    const xla::int64 pad_before = 0;

    padding[i] = {pad_before, pad_total - pad_before};
    rhs_dilation[i] = dims.spatial_dims[i].stride;
    window_strides[i] = dilations[dim];
  }

  // Redo the initial input padding.
  xla::PaddingConfig padding_config =
      XlaHelpers::MakeXlaPaddingConfig(XlaHelpers::I64List(padding_attr));

  xla::XlaBuilder* builder = grad_output.builder();
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp padded_input = xla::Pad(
      input,
      XlaHelpers::ScalarValue<float>(0, input_shape.element_type(), builder),
      padding_config);

  xla::PrecisionConfig precision_config =
      XlaHelpers::BuildPrecisionConfig(conv_precision);
  return xla::Transpose(xla::ConvGeneralDilated(
                            padded_input, grad_output, window_strides, padding,
                            /*lhs_dilation=*/ones, rhs_dilation, dnums,
                            /*feature_group_count=*/1,
                            /*batch_group_count=*/1, &precision_config),
                        {3, 2, 0, 1});
}

std::vector<std::pair<xla::int64, xla::int64>> MakePadding(
    tensorflow::gtl::ArraySlice<const xla::int64> padding) {
  std::vector<std::pair<xla::int64, xla::int64>> dims_padding;
  for (const auto dim_padding : padding) {
    dims_padding.emplace_back(dim_padding, dim_padding);
  }
  return dims_padding;
}

}  // namespace

xla::XlaOp BuildConvolution(
    const torch::jit::Node* node, const xla::XlaOp& input,
    const xla::XlaOp& kernel,
    const xla::PrecisionConfig::Precision conv_precision) {
  const auto stride = node->get<std::vector<int64_t>>(at::attr::stride).value();
  const auto padding =
      node->get<std::vector<int64_t>>(at::attr::padding).value();
  xla::PrecisionConfig precision_config =
      XlaHelpers::BuildPrecisionConfig(conv_precision);
  return BuildConvolution(input, kernel, XlaHelpers::I64List(stride),
                          XlaHelpers::I64List(padding), conv_precision);
}

xla::XlaOp BuildConvolution(
    const xla::XlaOp& input, const xla::XlaOp& kernel,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    const xla::PrecisionConfig::Precision conv_precision) {
  const auto dims_padding = MakePadding(padding);
  xla::PrecisionConfig precision_config =
      XlaHelpers::BuildPrecisionConfig(conv_precision);
  return xla::ConvWithGeneralPadding(
      input, kernel, stride, dims_padding,
      /*feature_group_count*/ 1, /*batch_group_count=*/1, &precision_config);
}

xla::XlaOp BuildConvolutionBias(
    const torch::jit::Node* node, const xla::XlaOp& input,
    const xla::XlaOp& kernel, const xla::XlaOp& bias,
    const xla::PrecisionConfig::Precision conv_precision) {
  const auto node_inputs = node->inputs();
  XLA_CHECK_GE(node_inputs.size(), size_t(4));
  const auto stride = node->get<std::vector<int64_t>>(at::attr::stride).value();
  const auto padding =
      node->get<std::vector<int64_t>>(at::attr::padding).value();
  return BuildConvolutionBias(input, kernel, bias, XlaHelpers::I64List(stride),
                              XlaHelpers::I64List(padding), conv_precision);
}

xla::XlaOp BuildConvolutionBias(
    const xla::XlaOp& input, const xla::XlaOp& kernel, const xla::XlaOp& bias,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    const xla::PrecisionConfig::Precision conv_precision) {
  xla::XlaOp conv =
      BuildConvolution(input, kernel, stride, padding, conv_precision);
  auto broadcast_sizes = XlaHelpers::SizesOfXlaOp(conv);
  XLA_CHECK_EQ(broadcast_sizes.size(), 4);
  // Remove the channels dimension.
  broadcast_sizes.erase(broadcast_sizes.begin() + 1);
  // Make the bias match the output dimensions.
  xla::XlaOp bias_broadcast =
      xla::Transpose(xla::Broadcast(bias, broadcast_sizes), {0, 3, 1, 2});
  return conv + bias_broadcast;
}

Conv2DGrads BuildConv2dBackward(
    const torch::jit::Node* node, const xla::XlaOp& grad_output,
    const xla::XlaOp& input, const xla::XlaOp& weight,
    const xla::PrecisionConfig::Precision conv_precision) {
  const auto stride = node->get<std::vector<int64_t>>(at::attr::stride).value();
  const auto padding =
      node->get<std::vector<int64_t>>(at::attr::padding).value();
  return BuildConv2dBackward(grad_output, input, weight,
                             XlaHelpers::I64List(stride),
                             XlaHelpers::I64List(padding), conv_precision);
}

Conv2DGrads BuildConv2dBackward(
    const xla::XlaOp& grad_output, const xla::XlaOp& input,
    const xla::XlaOp& weight,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    const xla::PrecisionConfig::Precision conv_precision) {
  xla::XlaOp grad_input = BuildThnnConv2dBackwardInput(
      grad_output, input, weight, stride, padding, conv_precision);
  xla::XlaOp grad_weight = BuildThnnConv2dBackwardWeight(
      grad_output, input, weight, stride, padding, conv_precision);
  xla::XlaBuilder* builder = grad_output.builder();
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp grad_bias = xla::Reduce(
      grad_output,
      XlaHelpers::ScalarValue<float>(0, input_shape.element_type(), builder),
      XlaHelpers::CreateAddComputation(input_shape.element_type()), {0, 2, 3});
  return {grad_input, grad_weight, grad_bias};
}

}  // namespace torch_xla
