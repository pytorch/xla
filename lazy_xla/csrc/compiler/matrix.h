#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

xla::XlaOp BuildDiagonal(xla::XlaOp input, xla::int64 offset, xla::int64 dim1,
                         xla::int64 dim2);

xla::XlaOp BuildDiagonalViewUpdate(xla::XlaOp target, xla::XlaOp input,
                                   xla::int64 offset, xla::int64 dim1,
                                   xla::int64 dim2);

}  // namespace torch_xla
