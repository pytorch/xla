#include "torch_xla/csrc/shape_helper.h"

#include "xla/hlo/builder/xla_builder.h"

#include "torch_xla/csrc/status.h"

namespace torch_xla {

const xla::Shape& ShapeHelper::ShapeOfXlaOp(xla::XlaOp op) {
  XLA_ASSIGN_OR_THROW(const xla::Shape* shape, GetShape(op));
  return *shape;
}

absl::StatusOr<const xla::Shape * absl_nonnull> GetShape(xla::XlaOp op) {
  return op.builder()->GetShapePtr(op);
}

}  // namespace torch_xla
