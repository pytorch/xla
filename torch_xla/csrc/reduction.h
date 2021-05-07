#pragma once

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

enum class ReductionMode {
  kNone,
  kMean,
  kSum,
};

xla::XlaOp BuildBinaryCrossEntropy(xla::XlaOp input, xla::XlaOp target,
                                   const absl::optional<xla::XlaOp>& weight,
                                   ReductionMode reduction);

xla::XlaOp BuildBinaryCrossEntropyBackward(
    xla::XlaOp grad_output, xla::XlaOp input, xla::XlaOp target,
    const absl::optional<xla::XlaOp>& weight, ReductionMode reduction);

xla::XlaOp BuildL1Loss(xla::XlaOp input, xla::XlaOp target,
                       ReductionMode reduction);

xla::XlaOp BuildL1LossBackward(xla::XlaOp grad_output, xla::XlaOp input,
                               xla::XlaOp target, ReductionMode reduction);

xla::XlaOp BuildMseLoss(xla::XlaOp input, xla::XlaOp target,
                        ReductionMode reduction);

xla::XlaOp BuildMseLossBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                xla::XlaOp target, ReductionMode reduction);

// Builds a mean by reducing all the dimensions listed in dimensions. If
// keep_reduced_dimensions is true, the reduced dimensions will be retained,
// with value 1.
xla::XlaOp BuildMean(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                     bool keep_reduced_dimensions);

xla::XlaOp BuildStdDeviation(xla::XlaOp input,
                             absl::Span<const xla::int64> dimensions,
                             bool keep_reduced_dimensions,
                             xla::int64 correction);

// Builds the sum of all values by reducing all the dimensions listed in
// dimensions. If keep_reduced_dimensions is true, the reduced dimensions will
// be retained, with value 1.
xla::XlaOp BuildSum(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                    bool keep_reduced_dimensions);

// Builds the max of all values by reducing in the given dimension. If
// keep_reduced_dimensions is true, the reduced dimension will be retained, with
// value 1.
xla::XlaOp BuildMaxInDim(xla::XlaOp input, xla::int64 dim,
                         bool keep_reduced_dimensions);

// Builds the max of all values by reducing in the given dimensions. If
// keep_reduced_dimensions is true, the reduced dimension will be retained, with
// value 1.
xla::XlaOp BuildMaxInDims(xla::XlaOp input,
                          absl::Span<const xla::int64> dimensions,
                          bool keep_reduced_dimensions);

// Builds the min of all values by reducing in the given dimension. If
// keep_reduced_dimensions is true, the reduced dimension will be retained, with
// value 1.
xla::XlaOp BuildMinInDim(xla::XlaOp input, xla::int64 dim,
                         bool keep_reduced_dimensions);

// Compute the indices of the maximum values of a tensor across a dimension.
xla::XlaOp BuildArgMax(xla::XlaOp input, xla::int64 dim, bool keepdim);

// Compute the indices of the minimum values of a tensor across a dimension.
xla::XlaOp BuildArgMin(xla::XlaOp input, xla::int64 dim, bool keepdim);

// Builds the product of all values by reducing all the dimensions listed in
// dimensions. If keep_reduced_dimensions is true, the reduced dimensions will
// be retained, with value 1.
xla::XlaOp BuildProd(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                     bool keep_reduced_dimensions);

// Compute the cumulative computation specified by "reducer" and "init" in the
// given dimension "dim".
xla::XlaOp BuildCumulativeComputation(xla::XlaOp input, xla::int64 dim,
                                      const xla::XlaComputation& reducer,
                                      xla::XlaOp init);

xla::XlaOp BuildAll(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                    bool keep_reduced_dimensions);

xla::XlaOp BuildAny(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                    bool keep_reduced_dimensions);

xla::XlaOp BuildVar(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                    xla::int64 correction, bool keep_reduced_dimensions);

xla::XlaOp BuildLogsumexp(xla::XlaOp input,
                          absl::Span<const xla::int64> dimensions,
                          bool keep_reduced_dimensions);

}  // namespace torch_xla
