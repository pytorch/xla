#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch_xla {

// Computes the convolution of the given input and kernel with the given
// precision, with the given stride and padding.
xla::XlaOp BuildConvolutionOverrideable(
    xla::XlaOp input, xla::XlaOp kernel,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    tensorflow::gtl::ArraySlice<const xla::int64> dilation, bool transposed,
    tensorflow::gtl::ArraySlice<const xla::int64> output_padding,
    xla::int64 groups);

// Same as above, then broadcasts the bias and adds it to the result.
xla::XlaOp BuildConvolutionOverrideableBias(
    xla::XlaOp input, xla::XlaOp kernel, xla::XlaOp bias,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    tensorflow::gtl::ArraySlice<const xla::int64> dilation, bool transposed,
    tensorflow::gtl::ArraySlice<const xla::int64> output_padding,
    xla::int64 groups);

struct ConvGrads {
  xla::XlaOp grad_input;
  xla::XlaOp grad_weight;
  xla::XlaOp grad_bias;
};

// Computes the gradients for a convolution with the given stride and padding.
ConvGrads BuildConvolutionBackwardOverrideable(
    xla::XlaOp grad_output, xla::XlaOp input, xla::XlaOp kernel,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    tensorflow::gtl::ArraySlice<const xla::int64> dilation, bool transposed,
    tensorflow::gtl::ArraySlice<const xla::int64> output_padding,
    xla::int64 groups);

}  // namespace torch_xla
