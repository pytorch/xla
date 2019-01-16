#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch_xla {
namespace ir {
namespace ops {

using LowerForShapeFn = std::function<xla::XlaOp(
    tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)>;

// Compute the output shape for the given input shapes and lowering.
xla::Shape InferOutputShape(
    tensorflow::gtl::ArraySlice<const xla::Shape> input_shapes,
    const LowerForShapeFn& core_lowering_fn);

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
