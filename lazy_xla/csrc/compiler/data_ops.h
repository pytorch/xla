#pragma once

#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "lazy_xla/csrc/compiler/xla_lowering_context.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

// Collection of XLA lowerings for operations which only involve some form of
// data movement and no computation.
namespace torch_lazy_tensors {

bool IsSparseGather(xla::XlaOp input, xla::XlaOp index, xla::int64 dim);

// Creates a new tensor with the same data as the input tensor and the specified
// output size.
xla::XlaOp BuildView(xla::XlaOp input,
                     absl::Span<const xla::int64> output_sizes);

// Squeezes the given dimension if trivial (size 1), returns the unchanged input
// otherwise.
xla::XlaOp SqueezeTrivialDimension(xla::XlaOp input, xla::int64 dim);

// Squeezes out the trivial (size 1) dimensions of the input.
xla::XlaOp SqueezeAllTrivialDimensions(xla::XlaOp input);

// Creates a new tensor with the singleton dimensions expanded to the specified
// output sizes.
xla::XlaOp BuildExpand(xla::XlaOp input,
                       absl::Span<const xla::int64> output_sizes);

// Insert a dimension of size one at the specified position.
xla::XlaOp BuildUnsqueeze(xla::XlaOp input, xla::int64 dim);

// Concatenates a list of tensors along a new dimension dim.
xla::XlaOp BuildStack(absl::Span<const xla::XlaOp> inputs, xla::int64 dim);

// Concatenates a list of tensors along an existing dimension specified by the
// dim argument.
xla::XlaOp BuildCat(absl::Span<const xla::XlaOp> inputs, xla::int64 dim);

// Splits a tensor into parts whose size is passed in split_sizes, along the dim
// dimension.
compiler::XlaOpVector BuildSplit(xla::XlaOp input,
                                 absl::Span<const xla::int64> split_sizes,
                                 xla::int64 dim);

// Creates an updated version of input, where, starting at base_indices, source
// if overlapped with input.
xla::XlaOp BuildUpdateSlice(xla::XlaOp input, xla::XlaOp source,
                            absl::Span<const xla::int64> base_indices);

xla::XlaOp BuildSlice(xla::XlaOp input,
                      absl::Span<const xla::int64> base_indices,
                      absl::Span<const xla::int64> sizes);

xla::XlaOp BoundIndices(xla::XlaOp index, xla::XlaOp max_index);

xla::XlaOp BuildResize(xla::XlaOp input, absl::Span<const xla::int64> size);

xla::XlaOp BuildUnselect(xla::XlaOp target, xla::XlaOp source, xla::int64 dim,
                         xla::int64 start, xla::int64 end, xla::int64 stride);

xla::XlaOp PadInDim(xla::XlaOp input, xla::int64 dim, xla::int64 pad_lo,
                    xla::int64 pad_hi, const xla::XlaOp* pad_value = nullptr);

}  // namespace torch_lazy_tensors
