#include "torch_xla/csrc/shape_helper.h"

#include "xla/client/xla_builder.h"
#include "third_party/xla_client/debug_macros.h"

namespace torch_xla {

const xla::Shape& ShapeHelper::ShapeOfXlaOp(xla::XlaOp op) {
  const xla::Shape* shape = ConsumeValue(op.builder()->GetShapePtr(op));
  return *shape;
}

}  // namespace torch_xla
