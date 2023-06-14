#ifndef XLA_TORCH_XLA_CSRC_CONVOLUTION_H_
#define XLA_TORCH_XLA_CSRC_CONVOLUTION_H_

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/tsl/platform/stringpiece.h" // StringPiece
#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h" // ConvOpAttrs
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


namespace torch_xla {

tsl::StatusOr<xla::XlaOp> PTXLAMakeXlaBackpropInputConvOp(
    StringPiece type_string, const xla::Shape& input_shape, xla::XlaOp filter,
    xla::XlaOp out_backprop, const ConvOpAttrs& attrs,
    xla::XlaOp* input_sizes = nullptr);

// Computes the convolution of the given input and kernel with the given
// precision, with the given stride and padding.
xla::XlaOp BuildConvolutionOverrideable(
    xla::XlaOp input, xla::XlaOp kernel, absl::Span<const int64_t> stride,
    absl::Span<const int64_t> padding, absl::Span<const int64_t> dilation,
    bool transposed, absl::Span<const int64_t> output_padding, int64_t groups);

// Same as above, then broadcasts the bias and adds it to the result.
xla::XlaOp BuildConvolutionOverrideableBias(
    xla::XlaOp input, xla::XlaOp kernel, xla::XlaOp bias,
    absl::Span<const int64_t> stride, absl::Span<const int64_t> padding,
    absl::Span<const int64_t> dilation, bool transposed,
    absl::Span<const int64_t> output_padding, int64_t groups);

struct ConvGrads {
  xla::XlaOp grad_input;
  xla::XlaOp grad_weight;
  xla::XlaOp grad_bias;
};

// Computes the gradients for a convolution with the given stride and padding.
ConvGrads BuildConvolutionBackwardOverrideable(
    xla::XlaOp grad_output, xla::XlaOp input, xla::XlaOp kernel,
    absl::Span<const int64_t> stride, absl::Span<const int64_t> padding,
    absl::Span<const int64_t> dilation, bool transposed,
    absl::Span<const int64_t> output_padding, int64_t groups);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_CONVOLUTION_H_