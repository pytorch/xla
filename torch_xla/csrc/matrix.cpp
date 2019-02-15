#include "torch_xla/csrc/matrix.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"

namespace torch_xla {

xla::XlaOp BuildTriu(const xla::XlaOp& input, int diagonal) {
  return xla::Select(xla::TriangleMask(input, diagonal - 1),
                     xla::ZerosLike(input), input);
}

xla::XlaOp BuildTril(const xla::XlaOp& input, int diagonal) {
  return xla::Select(xla::TriangleMask(input, diagonal), input,
                     xla::ZerosLike(input));
}

}  // namespace torch_xla
