#include "size_ops.h"
#include "helpers.h"
#include "tensorflow/compiler/xla/shape_util.h"
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

  const auto& sum_sizes = size_op_value_it->second;
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::int64 input_rank = input_shape.rank();

  XLA_CHECK_GE(input_rank, sum_sizes.size());

  xla::int64 rank_delta = input_rank - sum_sizes.size();
  std::vector<xla::int64> reduce_dimensions;
  for (xla::int64 i = 0; i < rank_delta; ++i) {
    reduce_dimensions.push_back(i);
  }
  for (xla::int64 i = rank_delta; i < input_rank; ++i) {
    xla::int64 input_dim = input_shape.dimensions(i);
    xla::int64 size_dim = sum_sizes[i - rank_delta];
    if (size_dim == 1 && input_dim > 1) {
      reduce_dimensions.push_back(i);
    }
  }
  if (reduce_dimensions.empty()) {
    return input;
  }
  return xla::Reduce(
      input,
      XlaHelpers::ScalarValue<float>(0, input_shape.element_type(),
                                     input.builder()),
      XlaHelpers::CreateAddComputation(input_shape.element_type()),
      reduce_dimensions);
}

}  // namespace jit
}  // namespace torch
