#include "reduction.h"
#include "helpers.h"

namespace torch {
namespace jit {

xla::XlaOp BuildSum(const Node* node, const xla::XlaOp& operand) {
  if (node->get<bool>(attr::keepdim).value()) {
    AT_ERROR("Sum with keepdim set not supported yet");
  }
  auto builder = operand.builder();
  const auto init_value = XlaHelpers::ScalarValue<float>(0, builder);
  const auto dimensions_to_reduce =
      node->get<std::vector<int64_t>>(attr::dim).value();
  return xla::Reduce(operand, init_value, XlaHelpers::CreateAddComputation(),
                     XlaHelpers::I64List(dimensions_to_reduce));
}

}  // namespace jit
}  // namespace torch
