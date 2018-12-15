#include "size_ops.h"
#include "helpers.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch {
namespace jit {

xla::XlaOp BuildSize(const Node* node, const xla::XlaOp& input,
                     std::vector<xla::int64>* size_op_result) {
  const auto shape_sizes = XlaHelpers::SizesOfXlaOp(input);
  *size_op_result = shape_sizes;
  auto builder = input.builder();
  return xla::ConstantR1<xla::int64>(builder, shape_sizes);
}

xla::XlaOp BuildSumToSize(
    const Node* node, const xla::XlaOp& input,
    const XlaComputationInOut::SizeOpValues& size_op_values_tracking) {
  const auto size_op_value_it =
      size_op_values_tracking.find(node->input(1)->unique());
  XLA_CHECK(size_op_value_it != size_op_values_tracking.end())
      << "prim::SumToSize only allowed when second parameter is a "
         "constant size: "
      << *node;
  const auto input_size = XlaHelpers::SizesOfXlaOp(input);
  XLA_CHECK_EQ(input_size, size_op_value_it->second)
      << "Only no-op prim::SumToSize supported for now";
  return input;
}

}  // namespace jit
}  // namespace torch
