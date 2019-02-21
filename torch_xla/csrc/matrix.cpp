#include "torch_xla/csrc/matrix.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {

xla::XlaOp BuildTriu(const xla::XlaOp& input, int diagonal) {
  return xla::Select(xla::TriangleMask(input, diagonal - 1),
                     xla::ZerosLike(input), input);
}

xla::XlaOp BuildTril(const xla::XlaOp& input, int diagonal) {
  return xla::Select(xla::TriangleMask(input, diagonal), input,
                     xla::ZerosLike(input));
}

xla::XlaOp BuildDiagonal(const xla::XlaOp& input, xla::int64 offset,
                         xla::int64 dim1, xla::int64 dim2) {
  auto input_shape = XlaHelpers::ShapeOfXlaOp(input);
  std::vector<xla::int64> dimension_permutation;
  for (xla::int64 dim = 0; dim < input_shape.rank(); ++dim) {
    if (dim != dim1 && dim != dim2) {
      dimension_permutation.push_back(dim);
    }
  }
  dimension_permutation.push_back(dim1);
  dimension_permutation.push_back(dim2);
  return xla::GetMatrixDiagonal(xla::Transpose(input, dimension_permutation),
                                offset);
}

}  // namespace torch_xla
