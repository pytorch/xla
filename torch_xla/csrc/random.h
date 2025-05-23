#ifndef XLA_TORCH_XLA_CSRC_RANDOM_H_
#define XLA_TORCH_XLA_CSRC_RANDOM_H_

#include "xla/hlo/builder/xla_builder.h"

namespace torch_xla {

// Set downcast to true if the caller knows the |maxval - minval| is appropriate
// for f16 dtype. We avoid computing the range on-the-fly since it incurs an XLA
// computation.
xla::XlaOp RngUniform(xla::XlaOp seed, const xla::Shape& shape,
                      xla::XlaOp minval, xla::XlaOp maxval,
                      bool downcast = false);

xla::XlaOp RngDiscreteUniform(xla::XlaOp seed, const xla::Shape& shape,
                              xla::XlaOp minval, xla::XlaOp maxval);

xla::XlaOp RngNormal(xla::XlaOp seed, const xla::Shape& shape, xla::XlaOp mean,
                     xla::XlaOp std);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_RANDOM_H_
