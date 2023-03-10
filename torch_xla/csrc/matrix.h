#ifndef XLA_TORCH_XLA_CSRC_MATRIX_H_
#define XLA_TORCH_XLA_CSRC_MATRIX_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

xla::XlaOp BuildTriu(xla::XlaOp input, int64_t diagonal);

xla::XlaOp BuildTril(xla::XlaOp input, int64_t diagonal);

xla::XlaOp BuildDiagonal(xla::XlaOp input, int64_t offset, int64_t dim1,
                         int64_t dim2);

xla::XlaOp BuildDiagonalViewUpdate(xla::XlaOp target, xla::XlaOp input,
                                   int64_t offset, int64_t dim1, int64_t dim2);

xla::XlaOp BuildInverse(xla::XlaOp input);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_MATRIX_H_