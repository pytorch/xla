#include "torch_xla/csrc/ir_util.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {

std::vector<const torch::lazy::Node*> Util::ComputePostOrder(
    const torch::lazy::Node* node, EmissionMap* emap) {
  std::vector<const torch::lazy::Node*> post_order;
  std::vector<const torch::lazy::Node*> queue;
  queue.push_back(node);
  while (!queue.empty()) {
    node = queue.back();
    auto it = emap->find(node);
    if (it == emap->end()) {
      (*emap)[node] = torch::lazy::Util::kEmitting;

      for (auto& output : node->operands()) {
        auto oit = emap->find(output.node);
        if (oit == emap->end()) {
          queue.push_back(output.node);
        } else if (oit->second == torch::lazy::Util::kEmitting) {
          XLA_ERROR() << "Graph loop found at " << *output.node;
        }
      }
    } else if (it->second == torch::lazy::Util::kEmitting) {
      for (auto& output : node->operands()) {
        auto oit = emap->find(output.node);
        XLA_CHECK(oit != emap->end() &&
                  oit->second == torch::lazy::Util::kEmitted)
            << "Graph loop found at " << *output.node;
      }
      (*emap)[node] = torch::lazy::Util::kEmitted;
      post_order.push_back(node);
      queue.pop_back();
    } else {
      XLA_CHECK_EQ(it->second, torch::lazy::Util::kEmitted);
      queue.pop_back();
    }
  }
  return post_order;
}

std::vector<const torch::lazy::Node*> Util::ComputePostOrder(
    absl::Span<const torch::lazy::Node* const> nodes, EmissionMap* emap) {
  std::vector<const torch::lazy::Node*> post_order;
  for (auto node : nodes) {
    auto node_post_order = ComputePostOrder(node, emap);
    post_order.insert(post_order.end(), node_post_order.begin(),
                      node_post_order.end());
  }
  return post_order;
}

std::vector<const torch::lazy::Node*> Util::ComputePostOrder(
    absl::Span<const torch::lazy::Node* const> nodes) {
  EmissionMap emap;
  return ComputePostOrder(nodes, &emap);
}

std::vector<torch::lazy::Value> Util::Clone(
    absl::Span<const torch::lazy::Value> values,
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
    absl::Span<const torch::lazy::Value> values) {
  std::vector<const torch::lazy::Node*> nodes;
  for (auto& value : values) {
    nodes.push_back(value.node.get());
  }
  std::vector<const torch::lazy::Node*> post_order = ComputePostOrder(nodes);
  return Clone(values, post_order);
}

size_t Util::GetGraphSize(absl::Span<const torch::lazy::Node* const> nodes) {
  std::vector<const torch::lazy::Node*> post_order = ComputePostOrder(nodes);
  return post_order.size();
}

}  // namespace torch_xla
