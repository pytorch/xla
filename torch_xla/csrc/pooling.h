#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch_xla {

// Computes max pooling for the given input.
xla::XlaOp BuildMaxPoolNd(
    xla::XlaOp input, xla::int64 spatial_dim_count,
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding, bool ceil_mode);

// Computes the gradient for max pooling.
xla::XlaOp BuildMaxPoolNdBackward(
    xla::XlaOp out_backprop, xla::XlaOp input, xla::int64 spatial_dim_count,
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding, bool ceil_mode);

// Computes average pooling for the given input.
xla::XlaOp BuildAvgPoolNd(
    xla::XlaOp input, xla::int64 spatial_dim_count,
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding, bool ceil_mode,
    bool count_include_pad);

// Computes the gradient for average pooling.
xla::XlaOp BuildAvgPoolNdBackward(
    xla::XlaOp out_backprop, xla::XlaOp input, xla::int64 spatial_dim_count,
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding, bool ceil_mode,
    bool count_include_pad);

// Computes adaptive average pooling for the given input and output size.
xla::XlaOp BuildAdaptiveAvgPool2d(
    xla::XlaOp input,
    tensorflow::gtl::ArraySlice<const xla::int64> output_size);

// Computes the gradient for adaptive average pooling.
xla::XlaOp BuildAdaptiveAvgPool2dBackward(xla::XlaOp out_backprop,
                                          xla::XlaOp input);

// Returns true if XLA lowering is supported for the given input and output size
// combination.
bool IsSupportedAdaptiveAvgPool2d(
    tensorflow::gtl::ArraySlice<const xla::int64> input_size,
    tensorflow::gtl::ArraySlice<const xla::int64> output_size);

}  // namespace torch_xla
