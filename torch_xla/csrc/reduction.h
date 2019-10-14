#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch_xla {

enum class ReductionMode {
  kNone,
  kMean,
  kSum,
};

xla::XlaOp BuildL1Loss(const xla::XlaOp& input, const xla::XlaOp& target,
                       ReductionMode reduction);

xla::XlaOp BuildL1LossBackward(const xla::XlaOp& grad_output,
                               const xla::XlaOp& input,
                               const xla::XlaOp& target,
                               ReductionMode reduction);

// Builds a mean by reducing all the dimensions listed in dimensions. If
// keep_reduced_dimensions is true, the reduced dimensions will be retained,
// with value 1.
xla::XlaOp BuildMean(const xla::XlaOp& input,
                     tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
                     bool keep_reduced_dimensions);

xla::XlaOp BuildStdDeviation(
    const xla::XlaOp& input,
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
    bool keep_reduced_dimensions, bool unbiased);

// Builds the sum of all values by reducing all the dimensions listed in
// dimensions. If keep_reduced_dimensions is true, the reduced dimensions will
// be retained, with value 1.
xla::XlaOp BuildSum(const xla::XlaOp& input,
                    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
                    bool keep_reduced_dimensions);

// Builds the max of all values by reducing in the given dimension. If
// keep_reduced_dimensions is true, the reduced dimension will be retained, with
// value 1.
xla::XlaOp BuildMaxInDim(const xla::XlaOp& input, xla::int64 dim,
                         bool keep_reduced_dimensions);

// Builds the min of all values by reducing in the given dimension. If
// keep_reduced_dimensions is true, the reduced dimension will be retained, with
// value 1.
xla::XlaOp BuildMinInDim(const xla::XlaOp& input, xla::int64 dim,
                         bool keep_reduced_dimensions);

// Compute the indices of the maximum values of a tensor across a dimension.
xla::XlaOp BuildArgMax(const xla::XlaOp& input, xla::int64 dim, bool keepdim);

// Compute the indices of the minimum values of a tensor across a dimension.
xla::XlaOp BuildArgMin(const xla::XlaOp& input, xla::int64 dim, bool keepdim);

// Builds the product of all values by reducing all the dimensions listed in
// dimensions. If keep_reduced_dimensions is true, the reduced dimensions will
// be retained, with value 1.
xla::XlaOp BuildProd(const xla::XlaOp& input,
                     tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
                     bool keep_reduced_dimensions);

// Compute the cumulative computation specified by "reducer" and "init" in the
// given dimension "dim".
xla::XlaOp BuildCumulativeComputation(const xla::XlaOp& input, xla::int64 dim,
                                      const xla::XlaComputation& reducer,
                                      const xla::XlaOp& init);

xla::XlaOp BuildAll(const xla::XlaOp& input,
                    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
                    bool keep_reduced_dimensions);

xla::XlaOp BuildAny(const xla::XlaOp& input,
                    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
                    bool keep_reduced_dimensions);

}  // namespace torch_xla
