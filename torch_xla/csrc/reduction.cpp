#include "torch_xla/csrc/reduction.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {

xla::XlaOp BuildSum(const torch::jit::Node* node, const xla::XlaOp& operand) {
  if (node->get<bool>(at::attr::keepdim).value()) {
    XLA_ERROR() << "Sum with keepdim set not supported yet";
  }
  xla::Shape operand_shape = XlaHelpers::ShapeOfXlaOp(operand);
  xla::XlaOp init_value = XlaHelpers::ScalarValue<float>(
      0, operand_shape.element_type(), operand.builder());
  const auto dimensions_to_reduce =
      node->get<std::vector<int64_t>>(at::attr::dim).value();
  return xla::Reduce(
      operand, init_value,
      XlaHelpers::CreateAddComputation(operand_shape.element_type()),
      XlaHelpers::I64List(dimensions_to_reduce));
}

}  // namespace torch_xla
