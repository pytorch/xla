#ifndef XLA_TORCH_XLA_CSRC_CONVOLUTION_HELPER_H_
#define XLA_TORCH_XLA_CSRC_CONVOLUTION_HELPER_H_

#include <string>
#include <string_view>

#include "absl/container/inlined_vector.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/xla_data.pb.h"

namespace torch_xla {

// -------------Convolution Helper Data Structure Start-------------------------
// Convolution helper Data Structures below are copied from TF2XLA bridge
// This part of helpers are origionally from
// https://github.com/tensorflow/tensorflow/blob/7f47eaf439d2b81de1aa24b10ed57eabd519dbdb/tensorflow/core/util/tensor_format.h

// Tensor format for input/output activations used in convolution operations.
// The mnemonics specify the meaning of each tensor dimension sorted from
// largest to smallest memory stride.
// N = Batch, H = Image Height, W = Image Width, C = Number of Channels.
enum TensorFormat {
  // FORMAT_NHWC is the default format.
  FORMAT_NHWC = 0,

  // FORMAT_NCHW often improves performance on GPUs.
  FORMAT_NCHW = 1,

  // NCHW_VECT_C is the most performant tensor format for cudnn6's quantized
  // int8 convolution and fused convolution. It is laid out in the same order
  // as NCHW, except that the size of the Channels dimension is divided by 4,
  // and a new dimension of size 4 is appended, which packs 4 adjacent channel
  // activations for the same pixel into an int32. Thus an NCHW format tensor
  // with dimensions [N, C, H, W] would have dimensions [N, C/4, H, W, 4] in
  // NCHW_VECT_C format.
  // A pre-condition of this format is that C must be a multiple of 4.
  FORMAT_NCHW_VECT_C = 2,

  // Similar to NHWC, but the size of the W dimension is divided by 4, and a
  // new dimension of size 4 is appended, which packs 4 adjacent activations
  // in the width dimension.
  FORMAT_NHWC_VECT_W = 3,

  // Note: although the current code in this file assumes VECT_C and VECT_W
  // enums imply int8x4 vectors, this should not be relied upon.
  // In the future we may change the meaning of these enums to include vectors
  // of other types such as int16x2, with op implementations automatically
  // determining which format is implied based on the datatype.

  // FORMAT_HWNC is for TPUs.
  FORMAT_HWNC = 4,

  // FORMAT_HWCN is for TPUs.
  FORMAT_HWCN = 5,
};

// This part of helpers are origionally from
// https://github.com/tensorflow/tensorflow/blob/7f47eaf439d2b81de1aa24b10ed57eabd519dbdb/tensorflow/core/util/padding.h

// Padding: the padding we apply to the input tensor along the rows and columns
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
enum Padding {
  VALID = 1,     // No padding.
  SAME = 2,      // Input and output layers have the same size.
  EXPLICIT = 3,  // Padding is explicitly specified
};

// This part of helpers are origionally from
// https://github.com/tensorflow/tensorflow/blob/31c35582c544e21c4b21b38ccc8e7299cfd08d6e/tensorflow/core/kernels/conv_grad_shape_utils.h

// Information about a single spatial dimension for a convolution
// backpropagation.
struct ConvBackpropSpatialDimension {
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

// Computed dimensions for a backwards convolution.
struct ConvBackpropDimensions {
  // Information about each spatial dimension.
  ::absl::InlinedVector<ConvBackpropSpatialDimension, 3> spatial_dims;

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
  int SpatialPadding(const Padding& padding, int dim) const;
};

// This part of helpers are origionally from
// https://github.com/tensorflow/tensorflow/blob/7f47eaf439d2b81de1aa24b10ed57eabd519dbdb/tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h

// ConvOpAttrs contains all of the metadata necessary to specify a TF or XLA
// convolution.
struct ConvOpAttrs {
  // Constructs a ConvOpAttrs, reading most of the attributes from `ctx`.
  // static StatusOr<ConvOpAttrs> Create(int num_spatial_dims, bool depthwise,
  //                                     OpKernelConstruction* ctx);

  bool depthwise;
  int num_spatial_dims;
  std::vector<tsl::int32> dilations;
  std::vector<tsl::int32> strides;
  Padding padding;
  std::vector<int64_t> explicit_paddings;
  TensorFormat data_format;
};

// -------------Convolution Helper Data Structure End-------------------------

// -------------Convolution Helper Function Start-------------------------
// Convolution helper functions below are copied from TF2XLA bridge
// This part of helpers are origionally from
// https://github.com/tensorflow/tensorflow/blob/7f47eaf439d2b81de1aa24b10ed57eabd519dbdb/tensorflow/core/util/tensor_format.h

// Returns the index of the batch dimension.
inline int GetTensorBatchDimIndex(int num_dims, TensorFormat format) {
  switch (format) {
    case FORMAT_NHWC:
    case FORMAT_NCHW:
    case FORMAT_NCHW_VECT_C:
    case FORMAT_NHWC_VECT_W:
      return 0;
    case FORMAT_HWNC:
      return num_dims - 2;
    case FORMAT_HWCN:
      return num_dims - 1;
    default:
      LOG(FATAL) << "Unknown format " << format;
      return -1;  // Avoid compiler warning about missing return value
  }
}

// Returns the index of the feature dimension. If format is NCHW_VECT_C, returns
// the index of the outer feature dimension (i.e. dimension 1, whose size would
// be num_features / 4 in this case).
inline int GetTensorFeatureDimIndex(int num_dims, TensorFormat format) {
  switch (format) {
    case FORMAT_NHWC:
    case FORMAT_HWNC:
      return num_dims - 1;
    case FORMAT_NHWC_VECT_W:
    case FORMAT_HWCN:
      return num_dims - 2;
    case FORMAT_NCHW:
    case FORMAT_NCHW_VECT_C:
      return 1;
    default:
      LOG(FATAL) << "Unknown format " << format;
      return -1;  // Avoid compiler warning about missing return value
  }
}

// Returns the number of spatial dims of a tensor of rank 'num_dims' and tensor
// format 'format'.
inline int GetTensorSpatialDims(int num_dims, TensorFormat format) {
  switch (format) {
    case FORMAT_NHWC:
    case FORMAT_NCHW:
    case FORMAT_HWNC:
    case FORMAT_HWCN:
      return num_dims - 2;  // Exclude N,C.
    case FORMAT_NCHW_VECT_C:
    case FORMAT_NHWC_VECT_W:
      // Note: the VECT_W is not counted as an independent spatial dim here,
      // since it just a component of the width dimension.
      return num_dims - 3;  // Exclude N,C,VectDim.
    default:
      LOG(FATAL) << "Unknown format " << format;
      return -1;  // Avoid compiler warning about missing return value
  }
}

// Convert a TensorFormat into string.
std::string ToString(TensorFormat format);

// Returns the dimension index of the specified 'spatial_dim' within an
// activation tensor. If format is NHWC_VECT_W and spatial_dim is 1, returns
// the index of the outer width dimension (i.e. dimension 2, whose size would
// be width / 4 in this case).
inline int GetTensorSpatialDimIndex(int num_dims, TensorFormat format,
                                    int spatial_dim) {
  CHECK(spatial_dim >= 0 &&
        spatial_dim < GetTensorSpatialDims(num_dims, format))
      << spatial_dim << " " << num_dims << " " << ToString(format);
  switch (format) {
    case FORMAT_NHWC:
    case FORMAT_NHWC_VECT_W:
      return spatial_dim + 1;
    case FORMAT_NCHW:
    case FORMAT_NCHW_VECT_C:
      return spatial_dim + 2;
    case FORMAT_HWNC:
    case FORMAT_HWCN:
      return spatial_dim;
    default:
      LOG(FATAL) << "Unknown format " << format;
      return -1;  // Avoid compiler warning about missing return value
  }
}

// This part of helpers are origionally from
// https://github.com/tensorflow/tensorflow/blob/7f47eaf439d2b81de1aa24b10ed57eabd519dbdb/tensorflow/core/kernels/conv_grad_shape_utils.h

// The V2 version computes the same outputs with arbitrary dilation rate and
// supports explicit padding.
tsl::Status ConvBackpropComputeDimensionsV2(
    std::string_view label, int num_spatial_dims, const xla::Shape& input_shape,
    const xla::Shape& filter_shape, const xla::Shape& out_backprop_shape,
    absl::Span<const tsl::int32> dilations,
    const std::vector<tsl::int32>& strides, Padding padding,
    TensorFormat data_format, ConvBackpropDimensions* dims,
    absl::Span<const int64_t> explicit_paddings);

// This part of helpers are origionally from
// https://github.com/tensorflow/tensorflow/blob/7f47eaf439d2b81de1aa24b10ed57eabd519dbdb/tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h

// Wrapper for ConvGeneralDilated with checking dims.
tsl::StatusOr<xla::XlaOp> MakeXlaBackpropInputConvOp(
    std::string_view type_string, const xla::Shape& input_shape,
    xla::XlaOp filter, xla::XlaOp out_backprop, const ConvOpAttrs& attrs,
    xla::XlaOp* input_sizes = nullptr);

// Wrapper for ConvGeneralDilated with checking dims.
tsl::StatusOr<xla::XlaOp> MakeXlaBackpropFilterConvOp(
    std::string_view type_string, xla::XlaOp activations,
    const xla::Shape& filter_shape, xla::XlaOp gradients,
    const ConvOpAttrs& attrs);

// -------------Convolution Helper Function End-------------------------

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_CONVOLUTION_HELPER_H_
