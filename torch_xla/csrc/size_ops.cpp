#include "size_ops.h"
#include "helpers.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch {
namespace jit {

xla::XlaOp BuildSize(
    const Node* node, const xla::XlaOp& input,
    XlaComputationInOut::SizeOpValues* size_op_values_tracking) {
  const auto shape_sizes = XlaHelpers::SizesOfXlaOp(input);
  const auto it_ok = size_op_values_tracking->emplace(
      std::pair<size_t, std::vector<xla::int64>>{node->output(0)->unique(),
                                                 shape_sizes});
  XLA_CHECK(it_ok.second);
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
         "constant size";
  const auto input_size = XlaHelpers::SizesOfXlaOp(input);
  XLA_CHECK_EQ(input_size, size_op_value_it->second)
      << "Only no-op prim::SumToSize supported for now";
  return input;
}

}  // namespace jit
}  // namespace torch
