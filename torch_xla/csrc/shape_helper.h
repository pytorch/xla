#ifndef XLA_TORCH_XLA_SHAPE_HELPER_H_
#define XLA_TORCH_XLA_SHAPE_HELPER_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

class ShapeHelper {
 public:
  // Returns the shape of the given XLA operation.
  static const xla::Shape& ShapeOfXlaOp(xla::XlaOp op);
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_SHAPE_HELPER_H_
