#ifndef XLA_TORCH_XLA_SHAPE_HELPER_H_
#define XLA_TORCH_XLA_SHAPE_HELPER_H_

#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "xla/hlo/builder/xla_builder.h"

namespace torch_xla {

class ShapeHelper {
 public:
  // Returns the shape of the given XLA operation.
  ABSL_DEPRECATED(
      "Use GetShape(op) instead. ShapeOfXlaOp() blindly "
      "de-references StatusOr returned by XLA, which is unsafe.")
  static const xla::Shape& ShapeOfXlaOp(xla::XlaOp op);
};

// Returns the shape of the given XLA operation.
absl::StatusOr<const xla::Shape * absl_nonnull> GetShape(xla::XlaOp op);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_SHAPE_HELPER_H_
