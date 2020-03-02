#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

xla::XlaOp RngUniform(xla::XlaOp seed, const xla::Shape& shape,
                      xla::XlaOp minval, xla::XlaOp maxval);

xla::XlaOp RngNormal(xla::XlaOp seed, const xla::Shape& shape, xla::XlaOp mean,
                     xla::XlaOp std);

}  // namespace torch_xla
