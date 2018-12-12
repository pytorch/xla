#include "pooling.h"
#include "helpers.h"
#include "tensorflow/compiler/xla/client/lib/pooling.h"
#include "torch/csrc/jit/autodiff.h"

namespace torch {
namespace jit {

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
PoolingOpAttributes Pooling2DOpAttributes(const Node* pooling_2d) {
  const auto kernel_size_attr = XlaHelpers::I64List(
      pooling_2d->get<std::vector<int64_t>>(attr::kernel_size).value());
  const auto stride_attr =
      pooling_2d->get<std::vector<int64_t>>(attr::stride).value();
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
      pooling_2d->get<std::vector<int64_t>>(attr::padding).value();
  CHECK_EQ(padding_attr.size(), 2);
  std::vector<std::pair<xla::int64, xla::int64>> padding;
  for (const xla::int64 dim_pad : padding_attr) {
    padding.push_back(std::make_pair(dim_pad, dim_pad));
  }
  return {kernel_size, stride, padding};
}

void CheckAvgPool2DIsSupported(const Node* node) {
  const auto node_inputs = node->inputs();
  CHECK_GE(node_inputs.size(), size_t(6));
  const auto ceil_mode = node->get<bool>(attr::ceil_mode).value();
  if (ceil_mode) {
    AT_ERROR("ceil_mode not supported for avg_pool2d yet");
  }
}

}  // namespace

xla::XlaOp BuildMaxPool2d(const Node* node, const xla::XlaOp& input) {
  const auto pooling_op_attributes = Pooling2DOpAttributes(node);
  auto builder = input.builder();
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  const auto init_value =
      xla::LiteralUtil::MinValue(input_shape.element_type());
  const auto xla_init_value = xla::ConstantLiteral(builder, init_value);
  const auto padding_config = XlaHelpers::MakeXlaPaddingConfig(
      node->get<std::vector<int64_t>>(attr::padding).value());
  const auto padded_input = xla::Pad(input, xla_init_value, padding_config);
  return xla::MaxPool(
      /*operand=*/padded_input,
      /*kernel_size=*/pooling_op_attributes.kernel_size,
      /*stride=*/pooling_op_attributes.stride,
      /*padding=*/xla::Padding::kValid,
      /*data_format=*/MakeNCHWFormat());
}

xla::XlaOp BuildMaxPool2dBackward(const Node* node,
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

xla::XlaOp BuildAvgPool2d(const Node* node, const xla::XlaOp& input) {
  // Inspired from tf2xla.
  CheckAvgPool2DIsSupported(node);
  const auto pooling_op_attributes = Pooling2DOpAttributes(node);
  const auto count_include_pad =
      node->get<bool>(attr::count_include_pad).value();
  return xla::AvgPool(
      /*operand=*/input,
      /*kernel_size=*/pooling_op_attributes.kernel_size,
      /*stride=*/pooling_op_attributes.stride,
      /*padding=*/pooling_op_attributes.padding,
      /*data_format=*/MakeNCHWFormat(),
      /*counts_include_padding=*/count_include_pad);
}

xla::XlaOp BuildAvgPool2dBackward(const Node* node,
                                  const xla::XlaOp& out_backprop,
                                  const xla::XlaOp& input) {
  // Inspired from tf2xla.
  CheckAvgPool2DIsSupported(node);
  const auto pooling_op_attributes = Pooling2DOpAttributes(node);
  auto gradients_size = XlaHelpers::ShapeSizes(XlaHelpers::ShapeOfXlaOp(input));
  const auto count_include_pad =
      node->get<bool>(attr::count_include_pad).value();

  return xla::AvgPoolGrad(
      /*out_backprop=*/out_backprop,
      /*gradients_size=*/gradients_size,
      /*kernel_size=*/pooling_op_attributes.kernel_size,
      /*stride=*/pooling_op_attributes.stride,
      /*spatial_padding=*/pooling_op_attributes.padding,
      /*data_format=*/MakeNCHWFormat(),
      /*counts_include_padding=*/count_include_pad);
}

}  // namespace jit
}  // namespace torch
