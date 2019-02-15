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

xla::XlaOp BuildMean(const xla::XlaOp& input,
                     tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
                     bool keep_reduced_dimensions) {
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp init_value = XlaHelpers::ScalarValue<float>(
      0, input_shape.element_type(), input.builder());
  xla::int64 element_count = 1;
  std::vector<xla::int64> new_dimensions;
  size_t idim = 0;
  for (xla::int64 i = 0; i < input_shape.rank(); ++i) {
    if (idim < dimensions.size() && dimensions[idim] == i) {
      element_count *= input_shape.dimensions(i);
      ++idim;
      if (keep_reduced_dimensions) {
        new_dimensions.push_back(1);
      }
    } else if (keep_reduced_dimensions) {
      new_dimensions.push_back(input_shape.dimensions(i));
    }
  }
  xla::XlaOp result = xla::Reduce(
      input, init_value,
      XlaHelpers::CreateAddComputation(input_shape.element_type()), dimensions);
  if (element_count > 1) {
    xla::XlaOp scale = XlaHelpers::ScalarValue<float>(
        1.0f / static_cast<float>(element_count), input_shape.element_type(),
        input.builder());
    result = xla::Mul(result, scale);
  }
  if (keep_reduced_dimensions) {
    result = xla::Reshape(result, new_dimensions);
  }
  return result;
}

}  // namespace torch_xla
