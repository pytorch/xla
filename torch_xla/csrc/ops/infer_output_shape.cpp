#include "torch_xla/csrc/ops/infer_output_shape.h"

#include "torch_xla/csrc/helpers.h"

namespace torch_xla {

xla::Shape InferOutputShape(absl::Span<const xla::Shape> input_shapes,
                            const LowerForShapeFn& core_lowering_fn) {
  xla::XlaBuilder b("InferOutputShape");
  std::vector<xla::XlaOp> parameters;
  for (size_t parameter_number = 0; parameter_number < input_shapes.size();
       ++parameter_number) {
    parameters.push_back(xla::Parameter(&b, parameter_number,
                                        input_shapes[parameter_number],
                                        absl::StrCat("p", parameter_number)));
  }
  xla::XlaOp result = core_lowering_fn(parameters);
  return XlaHelpers::ShapeOfXlaOp(result);
}

xla::Shape InferOutputShapes(absl::Span<const xla::Shape> input_shapes,
                             const LowerForShapesFn& core_lowering_fn) {
  xla::XlaBuilder b("InferOutputShape");
  std::vector<xla::XlaOp> parameters;
  for (size_t parameter_number = 0; parameter_number < input_shapes.size();
       ++parameter_number) {
    parameters.push_back(xla::Parameter(&b, parameter_number,
                                        input_shapes[parameter_number],
                                        absl::StrCat("p", parameter_number)));
  }
  std::vector<xla::XlaOp> results = core_lowering_fn(parameters);

  xla::Shape output_shape;
  if (results.size() == 2) {
    output_shape =
        xla::ShapeUtil::MakeTupleShape({XlaHelpers::ShapeOfXlaOp(results[0]),
                                        XlaHelpers::ShapeOfXlaOp(results[1])});
  } else {
    output_shape = XlaHelpers::ShapeOfXlaOp(results[0]);
  }
  return output_shape;
}

}  // namespace torch_xla
