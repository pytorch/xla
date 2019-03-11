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

size_t Util::GetGraphSize(
    tensorflow::gtl::ArraySlice<const Node* const> nodes) {
  std::vector<const Node*> post_order = ComputePostOrder(nodes);
  return post_order.size();
}

}  // namespace ir
}  // namespace torch_xla
