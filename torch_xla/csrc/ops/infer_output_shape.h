#ifndef XLA_TORCH_XLA_CSRC_OPS_INFER_OUTPUT_SHAPE_H_
#define XLA_TORCH_XLA_CSRC_OPS_INFER_OUTPUT_SHAPE_H_

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

using LowerForShapeFn =
    std::function<xla::XlaOp(absl::Span<const xla::XlaOp> operands)>;
using LowerForShapesFn = std::function<std::vector<xla::XlaOp>(
    absl::Span<const xla::XlaOp> operands)>;

// Compute the output shape for the given input shapes and lowering.
xla::Shape InferOutputShape(absl::Span<const xla::Shape> input_shapes,
                            const LowerForShapeFn& core_lowering_fn);

xla::Shape InferOutputShapes(absl::Span<const xla::Shape> input_shapes,
                             const LowerForShapesFn& core_lowering_fn);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_INFER_OUTPUT_SHAPE_H_