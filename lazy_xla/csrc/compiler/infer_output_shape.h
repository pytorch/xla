#pragma once

#include "absl/types/span.h"
#include "lazy_tensors/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

using LowerForShapeFn =
    std::function<xla::XlaOp(absl::Span<const xla::XlaOp> operands)>;

// Compute the output shape for the given input shapes and lowering.
lazy_tensors::Shape InferOutputShape(
    absl::Span<const lazy_tensors::Shape> input_shapes,
    const LowerForShapeFn& core_lowering_fn);

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
