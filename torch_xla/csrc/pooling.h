#ifndef XLA_TORCH_XLA_CSRC_POOLING_H_
#define XLA_TORCH_XLA_CSRC_POOLING_H_

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch_xla/csrc/device.h"

namespace torch_xla {

struct MaxPoolResult {
  xla::XlaOp result;
  xla::XlaOp indices;
};

// Computes max pooling for the given input.
MaxPoolResult BuildMaxPoolNd(xla::XlaOp input, int64_t spatial_dim_count,
                             absl::Span<const int64_t> kernel_size,
                             absl::Span<const int64_t> stride,
                             absl::Span<const int64_t> padding, bool ceil_mode);

// Computes the gradient for max pooling.
xla::XlaOp BuildMaxPoolNdBackward(xla::XlaOp out_backprop, xla::XlaOp input,
                                  int64_t spatial_dim_count,
                                  absl::Span<const int64_t> kernel_size,
                                  absl::Span<const int64_t> stride,
                                  absl::Span<const int64_t> padding,
                                  bool ceil_mode);

// Computes average pooling for the given input.
xla::XlaOp BuildAvgPoolNd(xla::XlaOp input, int64_t spatial_dim_count,
                          absl::Span<const int64_t> kernel_size,
                          absl::Span<const int64_t> stride,
                          absl::Span<const int64_t> padding, bool ceil_mode,
                          bool count_include_pad);

// Computes the gradient for average pooling.
xla::XlaOp BuildAvgPoolNdBackward(xla::XlaOp out_backprop, xla::XlaOp input,
                                  int64_t spatial_dim_count,
                                  absl::Span<const int64_t> kernel_size,
                                  absl::Span<const int64_t> stride,
                                  absl::Span<const int64_t> padding,
                                  bool ceil_mode, bool count_include_pad);

xla::XlaOp BuildMaxUnpoolNd(const torch::lazy::BackendDevice& device,
                            xla::XlaOp input, xla::XlaOp indices,
                            absl::Span<const int64_t> output_size);

xla::XlaOp BuildMaxUnpoolNdBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                    xla::XlaOp indices,
                                    absl::Span<const int64_t> output_size);

// Computes adaptive max pooling for the given input and output size.
MaxPoolResult BuildAdaptiveMaxPoolNd(xla::XlaOp input,
                                     absl::Span<const int64_t> output_size,
                                     int pool_dim);

// Computes the gradient for adaptive max pooling.
xla::XlaOp BuildAdaptiveMaxPoolNdBackward(xla::XlaOp out_backprop,
                                          xla::XlaOp input, int pool_dim);

// Computes adaptive average pooling for the given input and output size.
xla::XlaOp BuildAdaptiveAvgPool3d(xla::XlaOp input,
                                  absl::Span<const int64_t> output_size);

// Computes the gradient for adaptive average pooling.
xla::XlaOp BuildAdaptiveAvgPool3dBackward(xla::XlaOp out_backprop,
                                          xla::XlaOp input);

// Computes adaptive average pooling for the given input and output size.
xla::XlaOp BuildAdaptiveAvgPool2d(xla::XlaOp input,
                                  absl::Span<const int64_t> output_size);

// Computes the gradient for adaptive average pooling.
xla::XlaOp BuildAdaptiveAvgPool2dBackward(xla::XlaOp out_backprop,
                                          xla::XlaOp input);

// Returns true if XLA lowering is supported for the given input and output size
// combination.
bool IsSupportedAdaptivePool(absl::Span<const int64_t> input_size,
                             absl::Span<const int64_t> output_size,
                             int pool_dim);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_POOLING_H_