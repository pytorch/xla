#include "torch_xla/csrc/ir_util.h"

namespace torch_xla {

std::vector<torch::lazy::Value> Util::Clone(
    c10::ArrayRef<torch::lazy::Value> values,
    absl::Span<const torch::lazy::Node* const> post_order) {
  std::unordered_map<const torch::lazy::Node*, torch::lazy::NodePtr> clone_map;
  for (auto node : post_order) {
    if (clone_map.count(node) > 0) {
      continue;
    }
    std::vector<torch::lazy::Value> inputs;
    for (auto& output : node->operands()) {
      auto it = clone_map.find(output.node);
      XLA_CHECK(it != clone_map.end())
          << "Bad post-order: " << node->ToString();
      inputs.emplace_back(it->second, output.index);
    }
    const XlaNode* casted = dynamic_cast<const XlaNode*>(node);
    clone_map[node] = casted->Clone(inputs);
  }

  std::vector<torch::lazy::Value> cloned;
  for (auto& value : values) {
    auto it = clone_map.find(value.node.get());
    XLA_CHECK(it != clone_map.end()) << "Bad post-order: " << value->ToString();
    cloned.emplace_back(it->second, value.index);
  }
  return cloned;
}

std::vector<torch::lazy::Value> Util::Clone(
    c10::ArrayRef<torch::lazy::Value> values) {
  std::vector<const torch::lazy::Node*> nodes;
  for (auto& value : values) {
    nodes.push_back(value.node.get());
  }
  auto post_order = torch::lazy::Util::ComputePostOrder(nodes);
  return Clone(values, post_order);
}

}  // namespace torch_xla
