#include "lazy_xla/csrc/compiler/infer_output_shape.h"

#include "lazy_xla/csrc/compiler/helpers.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

lazy_tensors::Shape InferOutputShape(
    absl::Span<const lazy_tensors::Shape> input_shapes,
    const LowerForShapeFn& core_lowering_fn) {
  xla::XlaBuilder b("InferOutputShape");
  std::vector<xla::XlaOp> parameters;
  for (size_t parameter_number = 0; parameter_number < input_shapes.size();
       ++parameter_number) {
    parameters.push_back(xla::Parameter(
        &b, parameter_number,
        compiler::XlaHelpers::XlaShape(input_shapes[parameter_number]),
        absl::StrCat("p", parameter_number)));
  }
  xla::XlaOp result = core_lowering_fn(parameters);
  return compiler::XlaHelpers::LazyTensorsShape(
      compiler::XlaHelpers::ShapeOfXlaOp(result));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
