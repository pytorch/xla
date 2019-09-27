#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

xla::XlaOp BuildTriu(const xla::XlaOp& input, xla::int64 diagonal);

xla::XlaOp BuildTril(const xla::XlaOp& input, xla::int64 diagonal);

xla::XlaOp BuildDiagonal(const xla::XlaOp& input, xla::int64 offset,
                         xla::int64 dim1, xla::int64 dim2);

xla::XlaOp BuildDiagonalViewUpdate(const xla::XlaOp& target,
                                   const xla::XlaOp& input, xla::int64 offset,
                                   xla::int64 dim1, xla::int64 dim2);

}  // namespace torch_xla
