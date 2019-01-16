#include "ops/infer_output_shape.h"
#include "helpers.h"

namespace torch_xla {
namespace ir {
namespace ops {

xla::Shape InferOutputShape(
    tensorflow::gtl::ArraySlice<const xla::Shape> input_shapes,
    const LowerForShapeFn& core_lowering_fn) {
  xla::XlaBuilder b("infer_output_shape");
  std::vector<xla::XlaOp> parameters;
  for (size_t parameter_number = 0; parameter_number < input_shapes.size();
       ++parameter_number) {
    parameters.push_back(
        xla::Parameter(&b, parameter_number, input_shapes[parameter_number],
                       absl::StrCat("param_", parameter_number)));
  }
  xla::XlaOp result = core_lowering_fn(parameters);
  return XlaHelpers::ShapeOfXlaOp(result);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
