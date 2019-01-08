#include "pooling.h"
#include "helpers.h"
#include "tensorflow/compiler/xla/client/lib/pooling.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch/csrc/jit/autodiff.h"

namespace torch_xla {
namespace {

xla::TensorFormat MakeNCHWFormat() {
  return {/*batch_dimension=*/0,
          /*feature_dimension=*/1,
          /*spatial_dimensions=*/std::vector<xla::int64>{2, 3}};
}

// Holds the attributes common to all pooling operators.
struct PoolingOpAttributes {
  std::vector<xla::int64> kernel_size;
  std::vector<xla::int64> stride;
  std::vector<std::pair<xla::int64, xla::int64>> padding;
};

xla::XlaComputation CreateGeComputation(xla::PrimitiveType type) {
  xla::XlaBuilder reduction_builder("xla_ge_computation");
  const auto x = xla::Parameter(&reduction_builder, 0,
                                xla::ShapeUtil::MakeShape(type, {}), "x");
  const auto y = xla::Parameter(&reduction_builder, 1,
                                xla::ShapeUtil::MakeShape(type, {}), "y");
  xla::Ge(x, y);
  return reduction_builder.Build().ConsumeValueOrDie();
}

// Extract the pooling attributes for the given 2D pooling operator "node".
PoolingOpAttributes Pooling2DOpAttributes(const torch::jit::Node* pooling_2d) {
  const auto kernel_size_attr = XlaHelpers::I64List(
      pooling_2d->get<std::vector<int64_t>>(at::attr::kernel_size).value());
  const auto stride_attr =
      pooling_2d->get<std::vector<int64_t>>(at::attr::stride).value();
  // Create a NCHW kernel size with 1 for batch size and feature.
  std::vector<xla::int64> kernel_size(2, 1);
  kernel_size.insert(kernel_size.end(), kernel_size_attr.begin(),
                     kernel_size_attr.end());
  // Create a NCHW stride size with 1 for batch size and feature. Same as kernel
  // size if not specified.
  std::vector<xla::int64> stride;
  if (stride_attr.empty()) {
    stride = kernel_size;
  } else {
    stride.resize(2, 1);
    stride.insert(stride.end(), stride_attr.begin(), stride_attr.end());
  }
  const auto padding_attr =
      pooling_2d->get<std::vector<int64_t>>(at::attr::padding).value();
  XLA_CHECK_EQ(padding_attr.size(), 2);
  std::vector<std::pair<xla::int64, xla::int64>> padding;
  for (const xla::int64 dim_pad : padding_attr) {
    padding.push_back(std::make_pair(dim_pad, dim_pad));
  }
  return {kernel_size, stride, padding};
}

void CheckAvgPool2DIsSupported(const torch::jit::Node* node) {
  const auto node_inputs = node->inputs();
  XLA_CHECK_GE(node_inputs.size(), size_t(6));
  const auto ceil_mode = node->get<bool>(at::attr::ceil_mode).value();
  if (ceil_mode) {
    XLA_ERROR() << "ceil_mode not supported for avg_pool2d yet";
  }
}

// Compute the average pool kernel size required for the specified output_size
// from the given input_size, when the stride is the same as the kernel size.
std::vector<xla::int64> AdaptiveAvgPoolKernelSize(
    const std::vector<xla::int64>& input_size,
    const std::vector<xla::int64>& output_size) {
  // Create a NCHW kernel size with 1 for batch size and feature.
  std::vector<xla::int64> kernel_size(2, 1);
  for (int spatial_dim = 0; spatial_dim < 2; ++spatial_dim) {
    XLA_CHECK_EQ(input_size[2 + spatial_dim] % output_size[spatial_dim], 0)
        << "Target output size " << output_size[spatial_dim]
        << " doesn't divide the input size " << input_size[2 + spatial_dim];
    kernel_size.push_back(input_size[2 + spatial_dim] /
                          output_size[spatial_dim]);
  }
  return kernel_size;
}

}  // namespace

xla::XlaOp BuildMaxPool2d(const torch::jit::Node* node,
                          const xla::XlaOp& input) {
  const auto pooling_op_attributes = Pooling2DOpAttributes(node);
  auto builder = input.builder();
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  const auto init_value =
      xla::LiteralUtil::MinValue(input_shape.element_type());
  const auto xla_init_value = xla::ConstantLiteral(builder, init_value);
  const auto padding_config = XlaHelpers::MakeXlaPaddingConfig(
      node->get<std::vector<int64_t>>(at::attr::padding).value());
  const auto padded_input = xla::Pad(input, xla_init_value, padding_config);
  return xla::MaxPool(
      /*operand=*/padded_input,
      /*kernel_size=*/pooling_op_attributes.kernel_size,
      /*stride=*/pooling_op_attributes.stride,
      /*padding=*/xla::Padding::kValid,
      /*data_format=*/MakeNCHWFormat());
}

xla::XlaOp BuildMaxPool2dBackward(const torch::jit::Node* node,
                                  const xla::XlaOp& out_backprop,
                                  const xla::XlaOp& input) {
  auto builder = out_backprop.builder();
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  const auto init_value =
      XlaHelpers::ScalarValue<float>(0, input_shape.element_type(), builder);
  const auto select = CreateGeComputation(input_shape.element_type());
  const auto scatter =
      XlaHelpers::CreateAddComputation(input_shape.element_type());
  const auto pooling_op_attributes = Pooling2DOpAttributes(node);
  std::vector<std::pair<xla::int64, xla::int64>> window_padding;
  window_padding.resize(2);
  window_padding.insert(window_padding.end(),
                        pooling_op_attributes.padding.begin(),
                        pooling_op_attributes.padding.end());
  return xla::SelectAndScatterWithGeneralPadding(
      /*operand=*/input,
      /*select=*/select,
      /*window_dimensions=*/pooling_op_attributes.kernel_size,
      /*window_strides=*/pooling_op_attributes.stride,
      /*padding=*/window_padding,
      /*source=*/out_backprop,
      /*init_value=*/init_value,
      /*scatter=*/scatter);
}

xla::XlaOp BuildAvgPool2d(const torch::jit::Node* node,
                          const xla::XlaOp& input) {
  // Inspired from tf2xla.
  CheckAvgPool2DIsSupported(node);
  const auto pooling_op_attributes = Pooling2DOpAttributes(node);
  const auto count_include_pad =
      node->get<bool>(at::attr::count_include_pad).value();
  return xla::AvgPool(
      /*operand=*/input,
      /*kernel_size=*/pooling_op_attributes.kernel_size,
      /*stride=*/pooling_op_attributes.stride,
      /*padding=*/pooling_op_attributes.padding,
      /*data_format=*/MakeNCHWFormat(),
      /*counts_include_padding=*/count_include_pad);
}

xla::XlaOp BuildAvgPool2dBackward(const torch::jit::Node* node,
                                  const xla::XlaOp& out_backprop,
                                  const xla::XlaOp& input) {
  // Inspired from tf2xla.
  CheckAvgPool2DIsSupported(node);
  const auto pooling_op_attributes = Pooling2DOpAttributes(node);
  auto gradients_size = XlaHelpers::SizesOfXlaOp(input);
  const auto count_include_pad =
      node->get<bool>(at::attr::count_include_pad).value();

  return xla::AvgPoolGrad(
      /*out_backprop=*/out_backprop,
      /*gradients_size=*/gradients_size,
      /*kernel_size=*/pooling_op_attributes.kernel_size,
      /*stride=*/pooling_op_attributes.stride,
      /*spatial_padding=*/pooling_op_attributes.padding,
      /*data_format=*/MakeNCHWFormat(),
      /*counts_include_padding=*/count_include_pad);
}

xla::XlaOp BuildAdaptiveAvgPool2d(const torch::jit::Node* node,
                                  const xla::XlaOp& input) {
  const auto output_size = XlaHelpers::I64List(
      node->get<std::vector<int64_t>>(at::attr::output_size).value());
  XLA_CHECK_EQ(output_size.size(), 2) << "Invalid output size rank";
  const auto input_size = XlaHelpers::SizesOfXlaOp(input);
  XLA_CHECK_EQ(input_size.size(), 4) << "Only 4D tensors supported";
  const auto kernel_size = AdaptiveAvgPoolKernelSize(input_size, output_size);
  std::vector<std::pair<xla::int64, xla::int64>> no_padding(2);
  return xla::AvgPool(
      /*operand=*/input,
      /*kernel_size=*/kernel_size,
      /*stride=*/kernel_size,
      /*padding=*/no_padding,
      /*data_format=*/MakeNCHWFormat(),
      /*counts_include_padding=*/false);
}

xla::XlaOp BuildAdaptiveAvgPool2dBackward(const torch::jit::Node* node,
                                          const xla::XlaOp& out_backprop,
                                          const xla::XlaOp& input) {
  const auto out_backprop_size = XlaHelpers::SizesOfXlaOp(out_backprop);
  XLA_CHECK_EQ(out_backprop_size.size(), 4)
      << "Invalid rank of gradient output";
  std::vector<xla::int64> output_size{out_backprop_size[2],
                                      out_backprop_size[3]};
  const auto gradients_size = XlaHelpers::SizesOfXlaOp(input);
  XLA_CHECK_EQ(gradients_size.size(), 4) << "Only 4D tensors supported";
  const auto kernel_size =
      AdaptiveAvgPoolKernelSize(gradients_size, output_size);
  std::vector<std::pair<xla::int64, xla::int64>> no_padding(2);
  return xla::AvgPoolGrad(
      /*out_backprop=*/out_backprop,
      /*gradients_size=*/gradients_size,
      /*kernel_size=*/kernel_size,
      /*stride=*/kernel_size,
      /*spatial_padding=*/no_padding,
      /*data_format=*/MakeNCHWFormat(),
      /*counts_include_padding=*/false);
}

}  // namespace torch_xla
