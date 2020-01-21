#pragma once

#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

// Collection of XLA lowerings for operations which only involve some form of
// data movement and no computation.
namespace torch_xla {

bool IsSparseGather(xla::XlaOp input, xla::XlaOp index, xla::int64 dim);

// For input_sizes and a potentially incomplete output_sizes, return a complete
// output shape. The complete output shape has same total number of elements as
// input_sizes and matches output_sizes in all dimensions except for at most
// one, which can be inferred and stored as -1 in output_sizes.
std::vector<xla::int64> GetCompleteShape(
    tensorflow::gtl::ArraySlice<const xla::int64> output_sizes,
    tensorflow::gtl::ArraySlice<const xla::int64> input_sizes);

// Creates a new tensor with the same data as the input tensor and the specified
// output size.
xla::XlaOp BuildView(
    xla::XlaOp input,
    tensorflow::gtl::ArraySlice<const xla::int64> output_sizes);

// Squeezes the given dimension if trivial (size 1), returns the unchanged input
// otherwise.
xla::XlaOp SqueezeTrivialDimension(xla::XlaOp input, xla::int64 dim);

// Squeezes out the trivial (size 1) dimensions of the input.
xla::XlaOp SqueezeAllTrivialDimensions(xla::XlaOp input);

// Creates a new tensor with the singleton dimensions expanded to the specified
// output sizes.
xla::XlaOp BuildExpand(
    xla::XlaOp input,
    tensorflow::gtl::ArraySlice<const xla::int64> output_sizes);

std::vector<xla::int64> BuildSqueezedDimensions(
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
    xla::int64 squeeze_dim);

std::vector<xla::int64> BuildUnsqueezeDimensions(
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions, xla::int64 dim);

// Insert a dimension of size one at the specified position.
xla::XlaOp BuildUnsqueeze(xla::XlaOp input, xla::int64 dim);

// Concatenates a list of tensors along a new dimension dim.
xla::XlaOp BuildStack(tensorflow::gtl::ArraySlice<const xla::XlaOp> inputs,
                      xla::int64 dim);

// Concatenates a list of tensors along an existing dimension specified by the
// dim argument.
xla::XlaOp BuildCat(tensorflow::gtl::ArraySlice<const xla::XlaOp> inputs,
                    xla::int64 dim);

// Repeats the input tensor along each dimension by the given number of repeats.
xla::XlaOp BuildRepeat(xla::XlaOp input,
                       tensorflow::gtl::ArraySlice<const xla::int64> repeats);

// Creates an updated version of input, where, starting at base_indices, source
// if overlapped with input.
xla::XlaOp BuildUpdateSlice(
    xla::XlaOp input, xla::XlaOp source,
    tensorflow::gtl::ArraySlice<const xla::int64> base_indices);

xla::XlaOp BuildSlice(
    xla::XlaOp input,
    tensorflow::gtl::ArraySlice<const xla::int64> base_indices,
    tensorflow::gtl::ArraySlice<const xla::int64> sizes);

xla::XlaOp BoundIndices(xla::XlaOp index, xla::XlaOp max_index);

xla::XlaOp BuildTake(xla::XlaOp input, xla::XlaOp index);

xla::XlaOp BuildResize(xla::XlaOp input,
                       tensorflow::gtl::ArraySlice<const xla::int64> size);

xla::XlaOp BuildUnselect(xla::XlaOp target, xla::XlaOp source, xla::int64 dim,
                         xla::int64 start, xla::int64 end, xla::int64 stride);

xla::XlaOp BuildReflectionPad2d(
    xla::XlaOp input, tensorflow::gtl::ArraySlice<const xla::int64> padding);

xla::XlaOp BuildReflectionPad2dBackward(
    xla::XlaOp grad_output, xla::XlaOp input,
    tensorflow::gtl::ArraySlice<const xla::int64> padding);

xla::XlaOp PadInDim(xla::XlaOp input, xla::int64 dim, xla::int64 pad_lo,
                    xla::int64 pad_hi, const xla::XlaOp* pad_value = nullptr);

}  // namespace torch_xla
