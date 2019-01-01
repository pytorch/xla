#include "reduction.h"
#include "helpers.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch {
namespace jit {

xla::XlaOp BuildSum(const Node* node, const xla::XlaOp& operand) {
  if (node->get<bool>(attr::keepdim).value()) {
    XLA_ERROR() << "Sum with keepdim set not supported yet";
  }
  auto builder = operand.builder();
  xla::Shape operand_shape = XlaHelpers::ShapeOfXlaOp(operand);
  const auto init_value =
      XlaHelpers::ScalarValue<float>(0, operand_shape.element_type(), builder);
  const auto dimensions_to_reduce =
      node->get<std::vector<int64_t>>(attr::dim).value();
  return xla::Reduce(
      operand, init_value,
      XlaHelpers::CreateAddComputation(operand_shape.element_type()),
      XlaHelpers::I64List(dimensions_to_reduce));
}

}  // namespace jit
}  // namespace torch
