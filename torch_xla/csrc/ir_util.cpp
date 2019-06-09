#include "torch_xla/csrc/ir_util.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace ir {

std::vector<const Node*> Util::ComputePostOrder(const Node* node,
                                                EmissionMap* emap) {
  std::vector<const Node*> post_order;
  std::vector<const Node*> queue;
  queue.push_back(node);
  while (!queue.empty()) {
    node = queue.back();
    auto it = emap->find(node);
    if (it == emap->end()) {
      (*emap)[node] = kEmitting;

      for (auto& output : node->operands()) {
        auto oit = emap->find(output.node);
        if (oit == emap->end()) {
          queue.push_back(output.node);
        } else if (oit->second == kEmitting) {
          XLA_ERROR() << "Graph loop found at " << *output.node;
        }
      }
    } else if (it->second == kEmitting) {
      for (auto& output : node->operands()) {
        auto oit = emap->find(output.node);
        XLA_CHECK(oit != emap->end() && oit->second == kEmitted)
            << "Graph loop found at " << *output.node;
      }
      (*emap)[node] = kEmitted;
      post_order.push_back(node);
      queue.pop_back();
    } else {
      XLA_CHECK_EQ(it->second, kEmitted);
      queue.pop_back();
    }
  }
  return post_order;
}

std::vector<const Node*> Util::ComputePostOrder(
    tensorflow::gtl::ArraySlice<const Node* const> nodes) {
  EmissionMap emap;
  std::vector<const Node*> post_order;
  for (auto node : nodes) {
    auto node_post_order = ComputePostOrder(node, &emap);
    post_order.insert(post_order.end(), node_post_order.begin(),
                      node_post_order.end());
  }
  return post_order;
}

std::vector<Value> Util::Clone(
    tensorflow::gtl::ArraySlice<const Value> values,
    tensorflow::gtl::ArraySlice<const Node* const> post_order) {
  std::unordered_map<const Node*, NodePtr> clone_map;
  for (auto node : post_order) {
    if (clone_map.count(node) > 0) {
      continue;
    }
    std::vector<Value> inputs;
    for (auto& output : node->operands()) {
      auto it = clone_map.find(output.node);
      XLA_CHECK(it != clone_map.end())
          << "Bad post-order: " << node->ToString();
      inputs.emplace_back(it->second, output.index);
    }
    clone_map[node] = node->Clone(inputs);
  }

  std::vector<Value> cloned;
  for (auto& value : values) {
    auto it = clone_map.find(value.node.get());
    XLA_CHECK(it != clone_map.end()) << "Bad post-order: " << value->ToString();
    cloned.emplace_back(it->second, value.index);
  }
  return cloned;
}

std::vector<Value> Util::Clone(
    tensorflow::gtl::ArraySlice<const Value> values) {
  std::vector<const Node*> nodes;
  for (auto& value : values) {
    nodes.push_back(value.node.get());
  }
  std::vector<const Node*> post_order = ComputePostOrder(nodes);
  return Clone(values, post_order);
}

size_t Util::GetGraphSize(
    tensorflow::gtl::ArraySlice<const Node* const> nodes) {
  std::vector<const Node*> post_order = ComputePostOrder(nodes);
  return post_order.size();
}

}  // namespace ir
}  // namespace torch_xla
