#include "torch_xla/csrc/shape_helper.h"

#include "torch_xla/csrc/runtime/debug_macros.h"
#include "xla/hlo/builder/xla_builder.h"

namespace torch_xla {

const xla::Shape& ShapeHelper::ShapeOfXlaOp(xla::XlaOp op) {
  const xla::Shape* shape = ConsumeValue(op.builder()->GetShapePtr(op));
  return *shape;
}

}  // namespace torch_xla
