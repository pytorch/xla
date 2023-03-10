#ifndef XLA_TORCH_XLA_CSRC_REDUCTION_H_
#define XLA_TORCH_XLA_CSRC_REDUCTION_H_

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

enum class ReductionMode {
  kNone,
  kMean,
  kSum,
};

ReductionMode GetXlaReductionMode(int64_t reduction);

xla::XlaOp BuildBinaryCrossEntropy(xla::XlaOp input, xla::XlaOp target,
                                   const absl::optional<xla::XlaOp>& weight,
                                   ReductionMode reduction);

xla::XlaOp BuildBinaryCrossEntropyBackward(
    xla::XlaOp grad_output, xla::XlaOp input, xla::XlaOp target,
    const absl::optional<xla::XlaOp>& weight, ReductionMode reduction);

xla::XlaOp BuildMseLoss(xla::XlaOp input, xla::XlaOp target,
                        ReductionMode reduction);

xla::XlaOp BuildMseLossBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                xla::XlaOp target, ReductionMode reduction);

// Builds a mean by reducing all the dimensions listed in dimensions. If
// keep_reduced_dimensions is true, the reduced dimensions will be retained,
// with value 1.
xla::XlaOp BuildMean(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                     bool keep_reduced_dimensions);

xla::XlaOp BuildStdDeviation(xla::XlaOp input,
                             absl::Span<const int64_t> dimensions,
                             bool keep_reduced_dimensions, double correction);

// Builds the sum of all values by reducing all the dimensions listed in
// dimensions. If keep_reduced_dimensions is true, the reduced dimensions will
// be retained, with value 1.
xla::XlaOp BuildSum(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                    bool keep_reduced_dimensions);

// Builds the max of all values by reducing in the given dimension. If
// keep_reduced_dimensions is true, the reduced dimension will be retained, with
// value 1.
xla::XlaOp BuildMaxInDim(xla::XlaOp input, int64_t dim,
                         bool keep_reduced_dimensions);

// Builds the max of all values by reducing in the given dimensions. If
// keep_reduced_dimensions is true, the reduced dimension will be retained, with
// value 1.
xla::XlaOp BuildMaxInDims(xla::XlaOp input,
                          absl::Span<const int64_t> dimensions,
                          bool keep_reduced_dimensions);

// Builds the min of all values by reducing in the given dimension. If
// keep_reduced_dimensions is true, the reduced dimension will be retained, with
// value 1.
xla::XlaOp BuildMinInDim(xla::XlaOp input, int64_t dim,
                         bool keep_reduced_dimensions);

// Builds the min of all values by reducing in the given dimensions. If
// keep_reduced_dimensions is true, the reduced dimension will be retained, with
// value 1.
xla::XlaOp BuildMinInDims(xla::XlaOp input,
                          absl::Span<const int64_t> dimensions,
                          bool keep_reduced_dimensions);

// Compute the indices of the maximum values of a tensor across a dimension.
xla::XlaOp BuildArgMax(xla::XlaOp input, int64_t dim, bool keepdim);

// Compute the indices of the minimum values of a tensor across a dimension.
xla::XlaOp BuildArgMin(xla::XlaOp input, int64_t dim, bool keepdim);

// Builds the product of all values by reducing all the dimensions listed in
// dimensions. If keep_reduced_dimensions is true, the reduced dimensions will
// be retained, with value 1.
xla::XlaOp BuildProd(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                     bool keep_reduced_dimensions);

// Compute the cumulative computation specified by "reducer" and "init" in the
// given dimension "dim".
xla::XlaOp BuildCumulativeComputation(xla::XlaOp input, int64_t dim,
                                      const xla::XlaComputation& reducer,
                                      xla::XlaOp init);

xla::XlaOp BuildAll(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                    bool keep_reduced_dimensions);

xla::XlaOp BuildAny(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                    bool keep_reduced_dimensions);

xla::XlaOp BuildVar(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                    double correction, bool keep_reduced_dimensions);

xla::XlaOp BuildLogsumexp(xla::XlaOp input,
                          absl::Span<const int64_t> dimensions,
                          bool keep_reduced_dimensions);

xla::XlaOp BuildEinsum(absl::Span<const xla::XlaOp> operands,
                       const std::string& equation);

std::vector<xla::XlaOp> BuildEinsumBackward(const xla::XlaOp& grad_output,
                                            absl::Span<const xla::XlaOp> inputs,
                                            const std::string& equation);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_REDUCTION_H_