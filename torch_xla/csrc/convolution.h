#ifndef XLA_TORCH_XLA_CSRC_CONVOLUTION_H_
#define XLA_TORCH_XLA_CSRC_CONVOLUTION_H_

#include "absl/types/span.h"
// #include "tensorflow/core/lib/gtl/array_slice.h" // gtl::ArraySlice  // tensorflow::gtl::ArraySlice -> absl::Span<const T>
#include "tensorflow/compiler/xla/client/xla_builder.h"

// #include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h" // ConvOpAttrs
#include "tensorflow/core/util/tensor_format.h" // TensorFormat // GetTensorBatchDimIndex // GetTensorFeatureDimIndex // GetTensorSpatialDimIndex
// #include "tensorflow/core/kernels/conv_grad_shape_utils.h" // (done)ConvBackpropDimensions -> PTXLAConvBackpropDimensions // (done)ConvBackpropComputeDimensionsV2 -> PTXLAConvBackpropComputeDimensionsV2
// #include "tensorflow/core/util/padding.h" // tensorflow::Padding // 
#include "tensorflow/core/framework/tensor_shape.h" // TensorShape
// #include "tensorflow/compiler/tf2xla/shape_util.h" // XLAShapeToTensorShape

#include "tensorflow/tsl/lib/gtl/inlined_vector.h" // #include "tensorflow/core/lib/gtl/inlined_vector.h" // gtl::InlinedVector
#include "tensorflow/compiler/xla/xla_data.pb.h" // (done)ConvolutionDimensionNumbers // (done)PaddingType // (done)PrecisionConfig
#include "tensorflow/compiler/xla/client/xla_builder.h" // (done)DynamicConvInputGrad // (done)ConvGeneralDilated
#include "tensorflow/tsl/platform/stringpiece.h" // (done)StringPiece
#include "tensorflow/tsl/platform/errors.h" // (done)tsl::errors::InvalidArgument // 


namespace torch_xla {

// Convert an XLA Shape into the equivalent TensorFlow shape. May fail since
// not all XLA shapes can be represented as TensorShapes.
tsl::Status PTXLAXLAShapeToTensorShape(const xla::Shape& shape,
                             tensorflow::TensorShape* tensor_shape);

// Information about a single spatial dimension for a convolution
// backpropagation.
struct PTXLAConvBackpropSpatialDimension {
  int64_t input_size;
  int64_t filter_size;
  int64_t output_size;
  int64_t stride;
  int64_t dilation;

  // Output size after scaling by the stride.
  int64_t expanded_output_size;

  // Number of padding elements to be added before/after this dimension of
  // the input when computing Conv?DBackpropInput.
  int64_t pad_before, pad_after;
};

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

// Computed dimensions for a backwards convolution.
struct PTXLAConvBackpropDimensions {
  // Information about each spatial dimension.
  ::tsl::gtl::InlinedVector<PTXLAConvBackpropSpatialDimension, 3> spatial_dims;

  // Batch size.
  int64_t batch_size;

  // Input and output feature depth.
  int64_t in_depth, out_depth;

  // Convenience access methods for spatial dimensions properties.
  int64_t input_size(int dim) const { return spatial_dims[dim].input_size; }
  int64_t filter_size(int dim) const { return spatial_dims[dim].filter_size; }
  int64_t output_size(int dim) const { return spatial_dims[dim].output_size; }
  int64_t stride(int dim) const { return spatial_dims[dim].stride; }
  int64_t dilation(int dim) const { return spatial_dims[dim].dilation; }

  // Compute padding for the given spatial dimension.
  int SpatialPadding(const PTXLAPadding& padding, int dim) const;
};

// ConvOpAttrs contains all of the metadata necessary to specify a TF or XLA
// convolution.
struct PTXLAConvOpAttrs {
  // Constructs a ConvOpAttrs, reading most of the attributes from `ctx`.
  // static StatusOr<PTXLAConvOpAttrs> Create(int num_spatial_dims, bool depthwise,
  //                                     OpKernelConstruction* ctx);

  bool depthwise;
  int num_spatial_dims;
  std::vector<tsl::int32> dilations;
  std::vector<tsl::int32> strides;
  PTXLAPadding padding;
  std::vector<int64_t> explicit_paddings;
  tensorflow::TensorFormat data_format;
};

// The V2 version computes the same outputs with arbitrary dilation rate and
// supports explicit padding.
// TODO(b/67112639): Merge V2 versions and the original versions eventually.
tsl::Status PTXLAConvBackpropComputeDimensionsV2(
    tsl::StringPiece label, int num_spatial_dims, const tensorflow::TensorShape& input_shape,
    const tensorflow::TensorShape& filter_shape, const tensorflow::TensorShape& out_backprop_shape,
    const absl::Span<const tsl::int32>& dilations, const std::vector<tsl::int32>& strides,
    PTXLAPadding padding, absl::Span<const int64_t> explicit_paddings,
    tensorflow::TensorFormat data_format, PTXLAConvBackpropDimensions* dims);

tsl::StatusOr<xla::XlaOp> PTXLAMakeXlaBackpropInputConvOp(
    tsl::StringPiece type_string, const xla::Shape& input_shape, xla::XlaOp filter,
    xla::XlaOp out_backprop, const PTXLAConvOpAttrs& attrs,
    xla::XlaOp* input_sizes = nullptr);

tsl::StatusOr<xla::XlaOp> PTXLAMakeXlaBackpropFilterConvOp(tsl::StringPiece type_string,
                                                 xla::XlaOp activations,
                                                 const xla::Shape& filter_shape,
                                                 xla::XlaOp gradients,
                                                 const PTXLAConvOpAttrs& attrs);

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