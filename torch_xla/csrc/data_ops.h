#pragma once

#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

// Collection of XLA lowerings for operations which only involve some form of
// data movement and no computation.
namespace torch_xla {

struct DynamicReshapeInfo {
  xla::Shape output_shape;
  xla::int64 dynamic_dimension = -1;
};

bool IsSparseGather(const xla::XlaOp& input, const xla::XlaOp& index,
                    xla::int64 dim);

// For input_sizes and a potentially incomplete output_sizes, return a complete
// output shape. The complete output shape has same total number of elements as
// input_sizes and matches output_sizes in all dimensions except for at most
// one, which can be inferred and stored as -1 in output_sizes.
std::vector<xla::int64> GetCompleteShape(
    tensorflow::gtl::ArraySlice<const xla::int64> output_sizes,
    tensorflow::gtl::ArraySlice<const xla::int64> input_sizes);

absl::optional<DynamicReshapeInfo> GetDynamicReshapeInfo(
    const xla::Shape& input_shape,
    tensorflow::gtl::ArraySlice<const xla::int64> output_sizes);

// Creates a new tensor with the same data as the input tensor and the specified
// output size.
xla::XlaOp BuildView(
    const xla::XlaOp& input,
    tensorflow::gtl::ArraySlice<const xla::int64> output_sizes);

// Squeezes the given dimension if trivial (size 1), returns the unchanged input
// otherwise.
xla::XlaOp SqueezeTrivialDimension(const xla::XlaOp& input, size_t dim);

// Squeezes out the trivial (size 1) dimensions of the input.
xla::XlaOp SqueezeAllTrivialDimensions(const xla::XlaOp& input);

// Creates a new tensor with the singleton dimensions expanded to the specified
// output sizes.
xla::XlaOp BuildExpand(
    const xla::XlaOp& input,
    tensorflow::gtl::ArraySlice<const xla::int64> output_sizes);

std::vector<xla::int64> BuildUnsqueezeDimensions(
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions, size_t dim);

// Insert a dimension of size one at the specified position.
xla::XlaOp BuildUnsqueeze(const xla::XlaOp& input, size_t dim);

// Concatenates a list of tensors along a new dimension dim.
xla::XlaOp BuildStack(tensorflow::gtl::ArraySlice<const xla::XlaOp> inputs,
                      xla::int64 dim);

// Concatenates a list of tensors along an existing dimension specified by the
// dim argument.
xla::XlaOp BuildCat(tensorflow::gtl::ArraySlice<const xla::XlaOp> inputs,
                    xla::int64 dim);

// Repeats the input tensor along each dimension by the given number of repeats.
xla::XlaOp BuildRepeat(const xla::XlaOp& input,
                       tensorflow::gtl::ArraySlice<const xla::int64> repeats);

// Computes the number of splits with a dimension size and the split sizes.
size_t ComputeSplitCount(
    xla::int64 dim_size,
    tensorflow::gtl::ArraySlice<const xla::int64> split_sizes);

// Splits a tensor into parts whose size is passed in split_sizes, along the dim
// dimension.
std::vector<xla::XlaOp> BuildSplit(
    const xla::XlaOp& input,
    tensorflow::gtl::ArraySlice<const xla::int64> split_sizes, xla::int64 dim);

// Creates an updated version of input, where, starting at base_indices, source
// if overlapped with input.
xla::XlaOp BuildUpdateSlice(
    const xla::XlaOp& input, const xla::XlaOp& source,
    tensorflow::gtl::ArraySlice<const xla::int64> base_indices);

xla::XlaOp BuildSlice(
    const xla::XlaOp& input,
    tensorflow::gtl::ArraySlice<const xla::int64> base_indices,
    tensorflow::gtl::ArraySlice<const xla::int64> sizes);

xla::XlaOp BoundIndices(const xla::XlaOp& index, const xla::XlaOp& max_index);

xla::XlaOp BuildTake(const xla::XlaOp& input, const xla::XlaOp& index);

xla::XlaOp BuildResize(const xla::XlaOp& input,
                       tensorflow::gtl::ArraySlice<const xla::int64> size);

xla::XlaOp BuildUnselect(const xla::XlaOp& target, const xla::XlaOp& source,
                         xla::int64 dim, xla::int64 start, xla::int64 end,
                         xla::int64 stride);

}  // namespace torch_xla
