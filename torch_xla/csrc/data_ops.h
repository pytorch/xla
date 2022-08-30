#pragma once

#include <c10/core/ScalarType.h>

#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

// Collection of XLA lowerings for operations which only involve some form of
// data movement and no computation.
namespace torch_xla {

bool IsSparseGather(xla::XlaOp input, xla::XlaOp index, int64_t dim);

// For input_sizes and a potentially incomplete output_sizes, return a complete
// output shape. The complete output shape has same total number of elements as
// input_sizes and matches output_sizes in all dimensions except for at most
// one, which can be inferred and stored as -1 in output_sizes.
std::vector<int64_t> GetCompleteShape(absl::Span<const int64_t> output_sizes,
                                      absl::Span<const int64_t> input_sizes);

// Creates a new tensor with the same data as the input tensor and the specified
// output size.
xla::XlaOp BuildView(xla::XlaOp input, absl::Span<const int64_t> output_sizes);

// Squeezes the given dimension if trivial (size 1), returns the unchanged input
// otherwise.
xla::XlaOp SqueezeTrivialDimension(xla::XlaOp input, int64_t dim);

// Squeezes out the trivial (size 1) dimensions of the input.
xla::XlaOp SqueezeAllTrivialDimensions(xla::XlaOp input);

// Update Output Dynamic Dimensions based on input size()
xla::XlaOp SetDimensionSizes(xla::XlaOp input,
                             absl::Span<const xla::XlaOp> output_sizes);

// Creates a new tensor with the singleton dimensions expanded to the specified
// output sizes.
xla::XlaOp BuildExpand(xla::XlaOp input,
                       absl::Span<const int64_t> output_sizes);

std::vector<int64_t> BuildSqueezedDimensions(
    absl::Span<const int64_t> dimensions, int64_t squeeze_dim);

std::vector<int64_t> BuildUnsqueezeDimensions(
    absl::Span<const int64_t> dimensions, int64_t dim);

// Insert a dimension of size one at the specified position.
xla::XlaOp BuildUnsqueeze(xla::XlaOp input, int64_t dim);

// Concatenates a list of tensors along a new dimension dim.
xla::XlaOp BuildStack(absl::Span<const xla::XlaOp> inputs, int64_t dim);

// Concatenates a list of tensors along an existing dimension specified by the
// dim argument.
xla::XlaOp BuildCat(absl::Span<const xla::XlaOp> inputs, int64_t dim,
                    at::ScalarType dtype);

// Repeats the input tensor along each dimension by the given number of repeats.
xla::XlaOp BuildRepeat(xla::XlaOp input, absl::Span<const int64_t> repeats);

// Computes the number of splits with a dimension size and the split sizes.
size_t ComputeSplitCount(int64_t dim_size,
                         absl::Span<const int64_t> split_sizes);

// Splits a tensor into parts whose size is passed in split_sizes, along the dim
// dimension.
std::vector<xla::XlaOp> BuildSplit(xla::XlaOp input,
                                   absl::Span<const int64_t> split_sizes,
                                   int64_t dim);

// Creates an updated version of input, where, starting at base_indices, source
// if overlapped with input.
xla::XlaOp BuildUpdateSlice(xla::XlaOp input, xla::XlaOp source,
                            absl::Span<const int64_t> base_indices);

xla::XlaOp BuildSlice(xla::XlaOp input, absl::Span<const int64_t> base_indices,
                      absl::Span<const int64_t> sizes);

xla::XlaOp BoundIndices(xla::XlaOp index, xla::XlaOp max_index);

xla::XlaOp BuildTake(xla::XlaOp input, xla::XlaOp index);

xla::XlaOp BuildResize(xla::XlaOp input, absl::Span<const int64_t> size);

xla::XlaOp BuildUnselect(xla::XlaOp target, xla::XlaOp source, int64_t dim,
                         int64_t start, int64_t end, int64_t stride);

xla::XlaOp BuildReflectionPad2d(xla::XlaOp input,
                                absl::Span<const int64_t> padding);

xla::XlaOp BuildReflectionPadBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                      absl::Span<const int64_t> padding);

xla::XlaOp BuildReplicationPad(xla::XlaOp input,
                               absl::Span<const int64_t> padding);

xla::XlaOp BuildReplicationPadBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                       absl::Span<const int64_t> padding);

xla::XlaOp PadInDim(xla::XlaOp input, int64_t dim, int64_t pad_lo,
                    int64_t pad_hi, const xla::XlaOp* pad_value = nullptr);

}  // namespace torch_xla
