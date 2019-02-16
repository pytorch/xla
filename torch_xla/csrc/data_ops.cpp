#include "torch_xla/csrc/data_ops.h"

#include <functional>
#include <numeric>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace {

// Finds a at::prim::ListConstruct operation by id in the graph of "parent".
std::vector<const torch::jit::Value*> InputListAttr(
    const torch::jit::Node* parent, const size_t id) {
  const auto nodes = parent->owningGraph()->block()->nodes();
  std::vector<const torch::jit::Value*> result;
  for (const auto node : nodes) {
    if (node->kind() != at::prim::ListConstruct) {
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
  XLA_ERROR() << "Constant with id " << id << " not found";
}

}  // namespace

std::vector<xla::int64> GetCompleteShape(
    tensorflow::gtl::ArraySlice<const xla::int64> output_sizes,
    tensorflow::gtl::ArraySlice<const xla::int64> input_sizes) {
  c10::optional<size_t> incomplete_dim;
  int64_t incomplete_element_count = 1;
  for (size_t dim = 0; dim < output_sizes.size(); ++dim) {
    xla::int64 dim_size = output_sizes[dim];
    if (dim_size < 0) {
      XLA_CHECK(!incomplete_dim)
          << "More than one incomplete dimension found: " << *incomplete_dim
          << " and " << dim;
      incomplete_dim = dim;
    } else {
      incomplete_element_count *= dim_size;
    }
  }
  if (!incomplete_dim) {
    return std::vector<xla::int64>(output_sizes.begin(), output_sizes.end());
  }
  int64_t total_element_count =
      std::accumulate(input_sizes.begin(), input_sizes.end(), int64_t(1),
                      std::multiplies<int64_t>());
  XLA_CHECK_EQ(total_element_count % incomplete_element_count, 0)
      << "Cannot infer remaining dimension";
  std::vector<xla::int64> complete_output_sizes(output_sizes.begin(),
                                                output_sizes.end());
  complete_output_sizes[*incomplete_dim] =
      total_element_count / incomplete_element_count;
  return complete_output_sizes;
}

xla::XlaOp BuildView(const torch::jit::Node* node, const xla::XlaOp& input) {
  const auto node_inputs = node->inputs();
  XLA_CHECK_EQ(node_inputs.size(), 2);
  const auto node_outputs = node->outputs();
  XLA_CHECK_EQ(node_outputs.size(), 1);
  std::vector<int64_t> output_sizes =
      node->get<std::vector<int64_t>>(at::attr::size).value();
  return BuildView(input, XlaHelpers::I64List(output_sizes));
}

xla::XlaOp BuildView(
    const xla::XlaOp& input,
    tensorflow::gtl::ArraySlice<const xla::int64> output_sizes) {
  const auto complete_output_sizes =
      GetCompleteShape(output_sizes, XlaHelpers::SizesOfXlaOp(input));
  return xla::Reshape(input, complete_output_sizes);
}

xla::XlaOp BuildReshape(
    const torch::jit::Node* node, const xla::XlaOp& input,
    const std::unordered_map<size_t, std::vector<xla::int64>>&
        size_op_values_tracking) {
  std::vector<xla::int64> output_sizes;
  if (node->hasAttribute(at::attr::shape)) {
    output_sizes = XlaHelpers::I64List(
        node->get<std::vector<int64_t>>(at::attr::shape).value());
  } else {
    const auto size_op_value_it =
        size_op_values_tracking.find(node->input(1)->unique());
    XLA_CHECK(size_op_value_it != size_op_values_tracking.end())
        << "at::aten::reshape only allowed when second parameter is a "
           "constant size: "
        << *node;
    output_sizes = size_op_value_it->second;
  }
  return BuildView(input, output_sizes);
}

xla::XlaOp SqueezeTrivialDimension(const xla::XlaOp& input, size_t dim) {
  auto input_sizes = XlaHelpers::SizesOfXlaOp(input);
  XLA_CHECK_LT(dim, input_sizes.size());
  if (input_sizes[dim] != 1) {
    return input;
  }
  input_sizes.erase(input_sizes.begin() + dim);
  return xla::Reshape(input, input_sizes);
}

xla::XlaOp SqueezeAllTrivialDimensions(const xla::XlaOp& input) {
  auto input_sizes = XlaHelpers::SizesOfXlaOp(input);
  // Squeeze the trivial (of size 1) dimensions.
  std::vector<xla::int64> non_singleton_dimensions;
  std::copy_if(input_sizes.begin(), input_sizes.end(),
               std::back_inserter(non_singleton_dimensions),
               [](const size_t dim_size) { return dim_size != 1; });
  return xla::Reshape(input, non_singleton_dimensions);
}

xla::XlaOp BuildExpand(const torch::jit::Node* node, const xla::XlaOp& input) {
  const auto node_inputs = node->inputs();
  XLA_CHECK_GE(node_inputs.size(), 1);
  const auto node_outputs = node->outputs();
  XLA_CHECK_EQ(node_outputs.size(), 1);
  const auto output_sizes =
      node->get<std::vector<int64_t>>(at::attr::size).value();
  return BuildExpand(input, XlaHelpers::I64List(output_sizes));
}

xla::XlaOp BuildExpand(
    const xla::XlaOp& input,
    tensorflow::gtl::ArraySlice<const xla::int64> output_sizes) {
  auto input_sizes = XlaHelpers::SizesOfXlaOp(input);
  // Adjust the rank of the input to match the rank of the output.
  XLA_CHECK_LE(input_sizes.size(), output_sizes.size());
  for (size_t i = 0; i < output_sizes.size() - input_sizes.size(); ++i) {
    input_sizes.insert(input_sizes.begin(), 1);
  }
  xla::XlaOp implicit_reshape = xla::Reshape(input, input_sizes);
  xla::XlaOp squeezed_input = SqueezeAllTrivialDimensions(implicit_reshape);
  // Broadcast the squeezed tensor, the additional dimensions are to the left.
  std::vector<xla::int64> broadcast_sizes;
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    if (input_sizes[i] == 1) {
      broadcast_sizes.push_back(output_sizes[i]);
    }
  }
  xla::XlaOp broadcast = xla::Broadcast(squeezed_input, broadcast_sizes);
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

xla::XlaOp BuildUnsqueeze(const xla::XlaOp& input, size_t dim) {
  auto unsqueezed_sizes = XlaHelpers::SizesOfXlaOp(input);
  XLA_CHECK_LE(dim, unsqueezed_sizes.size());
  unsqueezed_sizes.insert(unsqueezed_sizes.begin() + dim, 1);
  return xla::Reshape(input, unsqueezed_sizes);
}

xla::XlaOp BuildStack(
    tensorflow::gtl::ArraySlice<const xla::XlaOp> inputs, xla::int64 dim) {
  // Reshape inputs along the dim axis.
  XLA_CHECK_GT(inputs.size(), 0);
  std::vector<xla::XlaOp> reshaped_inputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input_size = XlaHelpers::SizesOfXlaOp(inputs[i]);
    input_size.insert(input_size.begin() + dim, 1);
    reshaped_inputs.push_back(xla::Reshape(inputs[i], input_size));
  }
  return xla::ConcatInDim(inputs[0].builder(), reshaped_inputs, dim);
}

xla::XlaOp BuildStack(
    const torch::jit::Node* node,
    const std::function<xla::XlaOp(const torch::jit::Value*)>& node_op,
    xla::XlaBuilder* b) {
  const auto node_inputs = node->inputs();
  XLA_CHECK_EQ(node_inputs.size(), size_t(2));
  const auto stack_inputs = InputListAttr(node, node_inputs[0]->unique());
  const auto dim = node->get<int64_t>(at::attr::dim).value();
  std::vector<xla::XlaOp> inputs;
  for (size_t i = 0; i < stack_inputs.size(); ++i) {
    inputs.push_back(node_op(stack_inputs[i]));
  }
  return BuildStack(inputs, dim);
}

xla::XlaOp BuildCat(
    const torch::jit::Node* node,
    const std::function<xla::XlaOp(const torch::jit::Value*)>& node_op,
    xla::XlaBuilder* b) {
  const auto node_inputs = node->inputs();
  XLA_CHECK_EQ(node_inputs.size(), size_t(2));
  const auto stack_inputs = InputListAttr(node, node_inputs[0]->unique());
  const auto dim = node->get<int64_t>(at::attr::dim).value();
  std::vector<xla::XlaOp> cat_inputs;
  // Reshape inputs along the dim axis.
  for (size_t i = 0; i < stack_inputs.size(); ++i) {
    const auto stack_input = stack_inputs[i];
    cat_inputs.push_back(node_op(stack_input));
  }
  return xla::ConcatInDim(b, cat_inputs, dim);
}

std::vector<xla::XlaOp> BuildChunk(const torch::jit::Node* node,
                                   const xla::XlaOp& input) {
  int64_t chunks = node->get<int64_t>(at::attr::chunks).value();
  int64_t dim = node->get<int64_t>(at::attr::dim).value();
  XLA_CHECK_GE(dim, 0) << "Negative dimension specified for chunk operator";
  const auto input_sizes = XlaHelpers::SizesOfXlaOp(input);
  XLA_CHECK_LT(dim, input_sizes.size())
      << "Invalid dimension specified for chunk operator";
  int64_t size_in_dim = input_sizes[dim];
  int64_t split_size = (size_in_dim + chunks - 1) / chunks;
  std::vector<int64_t> split_sizes(chunks, split_size);
  split_sizes[chunks - 1] = split_size - (split_size * chunks - size_in_dim);
  std::vector<xla::XlaOp> splits(chunks);
  int64_t start_idx = 0;
  for (int64_t i = 0; i < chunks; ++i) {
    int64_t length = split_sizes[i];
    splits[i] = SliceInDim(input, start_idx, start_idx + length, 1, dim);
    start_idx += length;
  }
  return splits;
}

}  // namespace torch_xla
