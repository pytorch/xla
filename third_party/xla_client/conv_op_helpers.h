/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_CONV_OP_HELPERS_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_CONV_OP_HELPERS_H_

#include <vector>

#include "absl/strings/string_view.h"
#include "xla/client/xla_builder.h"
#include "xla/statusor.h"

// This header exposes utilities for translating TensorFlow convolution ops into
// XLA ops.

namespace xla {

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
enum ThreePadding {
  VALID = 1,     // No padding.
  SAME = 2,      // Input and output layers have the same size.
  EXPLICIT = 3,  // Padding is explicitly specified
};

// Tensor format for input/output activations used in convolution operations.
// The mnemonics specify the meaning of each tensor dimension sorted from
// largest to smallest memory stride.
// N = Batch, H = Image Height, W = Image Width, C = Number of Channels.
// TODO(pauldonnelly): It would probably be better to switch to a registration
// process for tensor formats, so specialized formats could be defined more
// locally to where they are used.
enum XLATensorFormat {
  // FORMAT_NHWC is the default format in TensorFlow.
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

// ConvOpAttrs contains all of the metadata necessary to specify an XLA
// convolution.
struct ConvOpAttrs {
//   // Constructs a ConvOpAttrs, reading most of the attributes from `ctx`.
//   static StatusOr<ConvOpAttrs> Create(int num_spatial_dims, bool depthwise,
//                                       OpKernelConstruction* ctx);
  bool depthwise;
  int num_spatial_dims;
  std::vector<tsl::int32> dilations;
  std::vector<tsl::int32> strides;
  ThreePadding padding;
  std::vector<tsl::int64> explicit_paddings;
  ConvolutionDimensionNumbers data_format; // or use TensorFormat from `https://github.com/tensorflow/tensorflow/blob/3fb52f183ff77b71e8de558b03ec92aa3011d447/tensorflow/core/util/tensor_format.h#L37`
  // ConvolutionDimensionNumbers is from `tensorflow/compiler/xla/xla_data.proto`
};

// Computes the convolution with the given input, filter and attributes. Errors
// returned by this function and the ones below are tagged with "type_string",
// which is the name of the TensorFlow operator using them.
StatusOr<XlaOp> MakeXlaForwardConvOp(
    absl::string_view type_string, XlaOp conv_input, XlaOp filter,
    const ConvOpAttrs& attrs,
    const PrecisionConfig* precision_config = nullptr);
// Computes the gradient with respect to the input, given the output gradient
// and the filter.
StatusOr<XlaOp> MakeXlaBackpropInputConvOp(
    absl::string_view type_string, const Shape& input_shape, XlaOp filter,
    XlaOp out_backprop, const ConvOpAttrs& attrs,
    const PrecisionConfig* precision_config = nullptr);
// Computes the gradient with respect to the filter, given the output gradient
// and the activations.
StatusOr<XlaOp> MakeXlaBackpropFilterConvOp(
    absl::string_view type_string, XlaOp activations, const Shape& filter_shape,
    XlaOp out_backprop, const ConvOpAttrs& attrs,
    const PrecisionConfig* precision_config = nullptr);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_CONV_OP_HELPERS_H_