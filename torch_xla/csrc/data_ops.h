#pragma once

#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

// Collection of XLA lowerings for operations which only involve some form of
// data movement and no computation.
namespace torch_xla {

bool IsSparseGather(xla::XlaOp input, xla::XlaOp index, xla::int64 dim);

// For input_sizes and a potentially incomplete output_sizes, return a complete
// output shape. The complete output shape has same total number of elements as
// input_sizes and matches output_sizes in all dimensions except for at most
// one, which can be inferred and stored as -1 in output_sizes.
std::vector<xla::int64> GetCompleteShape(
    absl::Span<const xla::int64> output_sizes,
    absl::Span<const xla::int64> input_sizes);

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

std::vector<xla::int64> BuildSqueezedDimensions(
    absl::Span<const xla::int64> dimensions, xla::int64 squeeze_dim);

std::vector<xla::int64> BuildUnsqueezeDimensions(
    absl::Span<const xla::int64> dimensions, xla::int64 dim);

// Insert a dimension of size one at the specified position.
xla::XlaOp BuildUnsqueeze(xla::XlaOp input, xla::int64 dim);

// Concatenates a list of tensors along a new dimension dim.
xla::XlaOp BuildStack(absl::Span<const xla::XlaOp> inputs, xla::int64 dim);

// Concatenates a list of tensors along an existing dimension specified by the
// dim argument.
xla::XlaOp BuildCat(absl::Span<const xla::XlaOp> inputs, xla::int64 dim);

// Repeats the input tensor along each dimension by the given number of repeats.
xla::XlaOp BuildRepeat(xla::XlaOp input, absl::Span<const xla::int64> repeats);

// Computes the number of splits with a dimension size and the split sizes.
size_t ComputeSplitCount(xla::int64 dim_size,
                         absl::Span<const xla::int64> split_sizes);

// Splits a tensor into parts whose size is passed in split_sizes, along the dim
// dimension.
std::vector<xla::XlaOp> BuildSplit(xla::XlaOp input,
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

xla::XlaOp BuildTake(xla::XlaOp input, xla::XlaOp index);

xla::XlaOp BuildResize(xla::XlaOp input, absl::Span<const xla::int64> size);

xla::XlaOp BuildUnselect(xla::XlaOp target, xla::XlaOp source, xla::int64 dim,
                         xla::int64 start, xla::int64 end, xla::int64 stride);

xla::XlaOp BuildReflectionPad2d(xla::XlaOp input,
                                absl::Span<const xla::int64> padding);

xla::XlaOp BuildReflectionPadBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                      absl::Span<const xla::int64> padding);

xla::XlaOp BuildReplicationPad(xla::XlaOp input,
                               absl::Span<const xla::int64> padding);

xla::XlaOp BuildReplicationPadBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                       absl::Span<const xla::int64> padding);

xla::XlaOp PadInDim(xla::XlaOp input, xla::int64 dim, xla::int64 pad_lo,
                    xla::int64 pad_hi, const xla::XlaOp* pad_value = nullptr);

}  // namespace torch_xla
