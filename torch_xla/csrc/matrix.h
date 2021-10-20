#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

xla::XlaOp BuildTriu(xla::XlaOp input, xla::int64_t diagonal);

xla::XlaOp BuildTril(xla::XlaOp input, xla::int64_t diagonal);

xla::XlaOp BuildDiagonal(xla::XlaOp input, xla::int64_t offset,
                         xla::int64_t dim1, xla::int64_t dim2);

xla::XlaOp BuildDiagonalViewUpdate(xla::XlaOp target, xla::XlaOp input,
                                   xla::int64_t offset, xla::int64_t dim1,
                                   xla::int64_t dim2);

xla::XlaOp BuildInverse(xla::XlaOp input);

}  // namespace torch_xla
