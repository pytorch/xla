#include "data_ops.h"
#include "helpers.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch {
namespace jit {

namespace {

// Graph nodes specify -1 for unknown dimensions. Return true iff all dimension
// sizes are positive.
bool IsCompleteShape(const std::vector<int64_t>& dim_sizes) {
  return std::all_of(dim_sizes.begin(), dim_sizes.end(),
                     [](const int64_t dim_size) { return dim_size >= 0; });
}

// Expand the input to the given output sizes.
xla::XlaOp BuildExpandToOutputSizes(
    const xla::XlaOp& input, const std::vector<xla::int64>& output_sizes) {
  auto input_sizes = XlaHelpers::ShapeSizes(XlaHelpers::ShapeOfXlaOp(input));
  // Adjust the rank of the input to match the rank of the output.
  XLA_CHECK_LE(input_sizes.size(), output_sizes.size());
  for (size_t i = 0; i < output_sizes.size() - input_sizes.size(); ++i) {
    input_sizes.insert(input_sizes.begin(), 1);
  }
  const auto implicit_reshape = xla::Reshape(input, input_sizes);
  // Squeeze the trivial (of size 1) dimensions.
  std::vector<xla::int64> non_singleton_dimensions;
  std::copy_if(input_sizes.begin(), input_sizes.end(),
               std::back_inserter(non_singleton_dimensions),
               [](const size_t dim_size) { return dim_size != 1; });
  const auto squeezed_input =
      xla::Reshape(implicit_reshape, non_singleton_dimensions);
  // Broadcast the squeezed tensor, the additional dimensions are to the left.
  std::vector<xla::int64> broadcast_sizes;
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    if (input_sizes[i] == 1) {
      broadcast_sizes.push_back(output_sizes[i]);
    }
  }
  const auto broadcast = xla::Broadcast(squeezed_input, broadcast_sizes);
  // Bring the dimensions added by broadcast where the trivial dimensions were.
  std::vector<xla::int64> reshape_permutation;
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    if (input_sizes[i] == 1) {
      reshape_permutation.push_back(i);
    }
  }
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    if (input_sizes[i] != 1) {
      reshape_permutation.push_back(i);
    }
  }
  return xla::Reshape(broadcast, reshape_permutation, output_sizes);
}

}  // namespace

xla::XlaOp BuildView(const Node* node, const xla::XlaOp& input) {
  const auto node_inputs = node->inputs();
  XLA_CHECK_EQ(node_inputs.size(), 2);
  const auto input_sizes = XlaHelpers::TensorDimensionSizes(node_inputs[0]);
  const auto node_outputs = node->outputs();
  XLA_CHECK_EQ(node_outputs.size(), 1);
  // Try to use the second argument of the operator as the target shape.
  std::vector<int64_t> output_sizes;
  switch (node->kind()) {
    case aten::view:
      output_sizes = node->get<std::vector<int64_t>>(attr::size).value();
      break;
    case aten::reshape:
      output_sizes = node->get<std::vector<int64_t>>(attr::shape).value();
      break;
    default:
      LOG(FATAL) << "Unexpected node kind, must be view or reshape";
  }
  // If the second argument doesn't fully specify the target shape, use the size
  // of the output.
  if (!IsCompleteShape(output_sizes)) {
    XLA_CHECK(node_outputs[0]->type()->cast<CompleteTensorType>());
    output_sizes = XlaHelpers::TensorDimensionSizes(node_outputs[0]);
  }
  JIT_ASSERTM(IsCompleteShape(output_sizes),
              "Cannot infer target size for aten::view");
  return xla::Reshape(input, XlaHelpers::I64List(output_sizes));
}

xla::XlaOp BuildExpand(const Node* node, const xla::XlaOp& input) {
  auto input_sizes = XlaHelpers::ShapeSizes(XlaHelpers::ShapeOfXlaOp(input));
  const auto node_outputs = node->outputs();
  XLA_CHECK_EQ(node_outputs.size(), 1);
  const auto output_sizes = XlaHelpers::TensorDimensionSizes(node_outputs[0]);
  return BuildExpandToOutputSizes(input, XlaHelpers::I64List(output_sizes));
}

xla::XlaOp BuildImplicitExpand(const xla::XlaOp& input,
                               const xla::XlaOp& output) {
  const auto output_sizes =
      XlaHelpers::ShapeSizes(XlaHelpers::ShapeOfXlaOp(output));
  const auto input_sizes =
      XlaHelpers::ShapeSizes(XlaHelpers::ShapeOfXlaOp(input));
  if (input_sizes.size() >= output_sizes.size() || input_sizes.empty()) {
    return input;
  }
  return BuildExpandToOutputSizes(input, output_sizes);
}

// Finds a prim::ListConstruct operation by id in the graph of "parent".
std::vector<const Value*> InputListAttr(const Node* parent, const size_t id) {
  const auto nodes = parent->owningGraph()->block()->nodes();
  std::vector<const Value*> result;
  for (const auto node : nodes) {
    if (node->kind() != prim::ListConstruct) {
      continue;
    }
    const auto node_outputs = node->outputs();
    XLA_CHECK_EQ(node_outputs.size(), size_t(1));
    const auto output = node_outputs[0];
    if (output->unique() != id) {
      continue;
    }
    const auto node_inputs = node->inputs();
    for (const auto input : node_inputs) {
      result.push_back(input);
    }
    return result;
  }
  XLA_CHECK(false) << "Constant with id " << id << " not found.";
}

xla::XlaOp BuildStack(const Node* node,
                      const std::function<xla::XlaOp(const Value*)>& node_op,
                      xla::XlaBuilder* b) {
  const auto node_inputs = node->inputs();
  XLA_CHECK_EQ(node_inputs.size(), size_t(2));
  const auto stack_inputs = InputListAttr(node, node_inputs[0]->unique());
  const auto dim = node->get<int64_t>(attr::dim).value();
  std::vector<xla::XlaOp> reshaped_inputs;
  // Reshape inputs along the dim axis.
  for (size_t i = 0; i < stack_inputs.size(); ++i) {
    auto reshaped_input_size =
        XlaHelpers::I64List(XlaHelpers::TensorDimensionSizes(stack_inputs[i]));
    reshaped_input_size.insert(reshaped_input_size.begin() + dim, 1);
    const auto stack_input = stack_inputs[i];
    reshaped_inputs.push_back(
        xla::Reshape(node_op(stack_input), reshaped_input_size));
  }
  return xla::ConcatInDim(b, reshaped_inputs, dim);
}

xla::XlaOp BuildCat(const Node* node,
                    const std::function<xla::XlaOp(const Value*)>& node_op,
                    xla::XlaBuilder* b) {
  const auto node_inputs = node->inputs();
  XLA_CHECK_EQ(node_inputs.size(), size_t(2));
  const auto stack_inputs = InputListAttr(node, node_inputs[0]->unique());
  const auto dim = node->get<int64_t>(attr::dim).value();
  std::vector<xla::XlaOp> cat_inputs;
  // Reshape inputs along the dim axis.
  for (size_t i = 0; i < stack_inputs.size(); ++i) {
    const auto stack_input = stack_inputs[i];
    cat_inputs.push_back(node_op(stack_input));
  }
  return xla::ConcatInDim(b, cat_inputs, dim);
}

std::vector<xla::XlaOp> BuildChunk(const Node* node, const xla::XlaOp& input) {
  const auto node_input = node->inputs()[0];
  int64_t chunks = node->get<int64_t>(attr::chunks).value();
  int64_t dim = node->get<int64_t>(attr::dim).value();
  int64_t size_in_dim = XlaHelpers::TensorDimensionSizes(node_input)[dim];
  int64_t split_size = (size_in_dim + chunks - 1) / chunks;
  std::vector<int64_t> split_sizes(chunks, split_size);
  split_sizes[chunks - 1] = split_size - (split_size * chunks - size_in_dim);
  std::vector<xla::XlaOp> splits(chunks);
  int64_t start_idx = 0;
  for (int64_t i = 0; i < chunks; ++i) {
    const auto length = split_sizes[i];
    splits[i] = SliceInDim(input, start_idx, start_idx + length, 1, dim);
    start_idx += length;
  }
  return splits;
}

}  // namespace jit
}  // namespace torch
