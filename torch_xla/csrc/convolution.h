#pragma once

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

// Computes the convolution of the given input and kernel with the given
// precision, with the given stride and padding.
xla::XlaOp BuildConvolutionOverrideable(
    xla::XlaOp input, xla::XlaOp kernel, absl::Span<const xla::int64> stride,
    absl::Span<const xla::int64> padding, absl::Span<const xla::int64> dilation,
    bool transposed, absl::Span<const xla::int64> output_padding,
    xla::int64 groups);

// Same as above, then broadcasts the bias and adds it to the result.
xla::XlaOp BuildConvolutionOverrideableBias(
    xla::XlaOp input, xla::XlaOp kernel, xla::XlaOp bias,
    absl::Span<const xla::int64> stride, absl::Span<const xla::int64> padding,
    absl::Span<const xla::int64> dilation, bool transposed,
    absl::Span<const xla::int64> output_padding, xla::int64 groups);

struct ConvGrads {
  xla::XlaOp grad_input;
  xla::XlaOp grad_weight;
  xla::XlaOp grad_bias;
};

// Computes the gradients for a convolution with the given stride and padding.
ConvGrads BuildConvolutionBackwardOverrideable(
    xla::XlaOp grad_output, xla::XlaOp input, xla::XlaOp kernel,
    absl::Span<const xla::int64> stride, absl::Span<const xla::int64> padding,
    absl::Span<const xla::int64> dilation, bool transposed,
    absl::Span<const xla::int64> output_padding, xla::int64 groups);

}  // namespace torch_xla
