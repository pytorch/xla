#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

xla::XlaOp RngUniform(xla::XlaOp seed, const xla::Shape& shape,
                      xla::XlaOp minval, xla::XlaOp maxval);

xla::XlaOp RngDiscreteUniform(xla::XlaOp seed, const xla::Shape& shape,
                              xla::XlaOp minval, xla::XlaOp maxval);

xla::XlaOp RngNormal(xla::XlaOp seed, const xla::Shape& shape, xla::XlaOp mean,
                     xla::XlaOp std);

xla::XlaOp BuildRandpermOut(int64_t n, xla::XlaBuilder* builder);

}  // namespace torch_xla
