#include "torch_xla/csrc/size_ops.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {

xla::XlaOp BuildSize(const torch::jit::Node* node, const xla::XlaOp& input,
                     std::vector<xla::int64>* size_op_result) {
  const auto shape_sizes = XlaHelpers::SizesOfXlaOp(input);
  *size_op_result = shape_sizes;
  xla::XlaBuilder* builder = input.builder();
  return xla::ConstantR1<xla::int64>(builder, shape_sizes);
}

}  // namespace torch_xla
