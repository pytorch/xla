#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch_xla {

// Builds a mean by reducing all the dimensions listed in dimensions. If
// keep_reduced_dimensions is true, the reduced dimensions will be retained,
// with value 1.
xla::XlaOp BuildMean(const xla::XlaOp& input,
                     tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
                     bool keep_reduced_dimensions);

// Builds the sum of all values by reducing all the dimensions listed in
// dimensions. If keep_reduced_dimensions is true, the reduced dimensions will
// be retained, with value 1.
xla::XlaOp BuildSum(const xla::XlaOp& input,
                    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
                    bool keep_reduced_dimensions);

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
