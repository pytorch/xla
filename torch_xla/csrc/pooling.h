#pragma once

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch_xla/csrc/device.h"

namespace torch_xla {

struct MaxPoolResult {
  xla::XlaOp result;
  xla::XlaOp indices;
};

// Computes max pooling for the given input.
MaxPoolResult BuildMaxPoolNd(xla::XlaOp input, xla::int64 spatial_dim_count,
                             absl::Span<const xla::int64> kernel_size,
                             absl::Span<const xla::int64> stride,
                             absl::Span<const xla::int64> padding,
                             bool ceil_mode);

// Computes the gradient for max pooling.
xla::XlaOp BuildMaxPoolNdBackward(xla::XlaOp out_backprop, xla::XlaOp input,
                                  xla::int64 spatial_dim_count,
                                  absl::Span<const xla::int64> kernel_size,
                                  absl::Span<const xla::int64> stride,
                                  absl::Span<const xla::int64> padding,
                                  bool ceil_mode);

// Computes average pooling for the given input.
xla::XlaOp BuildAvgPoolNd(xla::XlaOp input, xla::int64 spatial_dim_count,
                          absl::Span<const xla::int64> kernel_size,
                          absl::Span<const xla::int64> stride,
                          absl::Span<const xla::int64> padding, bool ceil_mode,
                          bool count_include_pad);

// Computes the gradient for average pooling.
xla::XlaOp BuildAvgPoolNdBackward(xla::XlaOp out_backprop, xla::XlaOp input,
                                  xla::int64 spatial_dim_count,
                                  absl::Span<const xla::int64> kernel_size,
                                  absl::Span<const xla::int64> stride,
                                  absl::Span<const xla::int64> padding,
                                  bool ceil_mode, bool count_include_pad);

xla::XlaOp BuildMaxUnpoolNd(const Device& device, xla::XlaOp input,
                            xla::XlaOp indices,
                            absl::Span<const xla::int64> output_size);

xla::XlaOp BuildMaxUnpoolNdBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                    xla::XlaOp indices,
                                    absl::Span<const xla::int64> output_size);

// Computes adaptive average pooling for the given input and output size.
xla::XlaOp BuildAdaptiveAvgPool2d(xla::XlaOp input,
                                  absl::Span<const xla::int64> output_size);

// Computes the gradient for adaptive average pooling.
xla::XlaOp BuildAdaptiveAvgPool2dBackward(xla::XlaOp out_backprop,
                                          xla::XlaOp input);

// Returns true if XLA lowering is supported for the given input and output size
// combination.
bool IsSupportedAdaptiveAvgPool2d(absl::Span<const xla::int64> input_size,
                                  absl::Span<const xla::int64> output_size);

}  // namespace torch_xla
