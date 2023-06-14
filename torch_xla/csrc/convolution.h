#ifndef XLA_TORCH_XLA_CSRC_CONVOLUTION_H_
#define XLA_TORCH_XLA_CSRC_CONVOLUTION_H_

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h" // ConvOpAttrs
#include "tensorflow/core/util/tensor_format.h" // GetTensorBatchDimIndex // GetTensorFeatureDimIndex // GetTensorSpatialDimIndex
#include "tensorflow/core/kernels/conv_grad_shape_utils.h" // ConvBackpropDimensions // 
// #include "tensorflow/core/util/padding.h" // tensorflow::Padding // 
#include "tensorflow/core/util/tensor_format.h" // TensorFormat
#include "tensorflow/core/framework/tensor_shape.h" // TensorShape
#include "tensorflow/compiler/tf2xla/shape_util.h" // XLAShapeToTensorShape
#include "tensorflow/core/kernels/conv_grad_shape_utils.h" // ConvBackpropComputeDimensionsV2

#include "tensorflow/compiler/xla/xla_data.pb.h" // (done)ConvolutionDimensionNumbers // (done)PaddingType // (done)PrecisionConfig
#include "tensorflow/compiler/xla/client/xla_builder.h" // (done)DynamicConvInputGrad // (done)ConvGeneralDilated
#include "tensorflow/tsl/platform/stringpiece.h" // (done)StringPiece
#include "tensorflow/tsl/platform/errors.h" // (done)tsl::errors::InvalidArgument // 


namespace torch_xla {

// PTXLAPadding: the padding we apply to the input tensor along the rows and columns
// dimensions. This is usually used to make sure that the spatial dimensions do
// not shrink when we progress with convolutions. Three types of padding are
// supported:
//   VALID: No padding is carried out.
//   SAME: The pad value is computed so that the output will have the same
//         dimensions as the input.
//   EXPLICIT: The user specifies the pad values in the explicit_paddings
//             attribute.
// The padded area is typically zero-filled. For pooling ops, the padded area is
// instead ignored. For max pool, this is equivalent to padding with -infinity.
enum PTXLAPadding {
  VALID = 1,     // No padding.
  SAME = 2,      // Input and output layers have the same size.
  EXPLICIT = 3,  // PTXLAPadding is explicitly specified
};

// ConvOpAttrs contains all of the metadata necessary to specify a TF or XLA
// convolution.
struct PTXLAConvOpAttrs {
  // Constructs a ConvOpAttrs, reading most of the attributes from `ctx`.
  // static StatusOr<PTXLAConvOpAttrs> Create(int num_spatial_dims, bool depthwise,
  //                                     OpKernelConstruction* ctx);

  bool depthwise;
  int num_spatial_dims;
  std::vector<int32> dilations;
  std::vector<int32> strides;
  PTXLAPadding padding;
  std::vector<int64_t> explicit_paddings;
  tensorflow::TensorFormat data_format;
};

tsl::StatusOr<xla::XlaOp> PTXLAMakeXlaBackpropInputConvOp(
    tsl::StringPiece type_string, const xla::Shape& input_shape, xla::XlaOp filter,
    xla::XlaOp out_backprop, const PTXLAConvOpAttrs& attrs,
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