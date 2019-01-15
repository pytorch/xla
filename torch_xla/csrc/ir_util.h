#pragma once

#include <unordered_map>
#include <vector>

#include "ir.h"

namespace torch_xla {
namespace ir {

class Util {
 public:
  // Tracks the emission status of the nodes during the post-order generation.
  // It helps tracking loops within the computation graphs.
  enum EmitStatus {
    kNotEmitted,
    kEmitting,
    kEmitted,
  };

  using EmissionMap = std::unordered_map<const Node*, EmitStatus>;

  // Calculates the post-order necessary to lower the given node. The returned
  // post-order can be empty if the node has already been lowered.
  static std::vector<const Node*> ComputePostOrder(const Node* node,
                                                   EmissionMap* emap);

  static std::vector<const Node*> ComputePostOrder(const Node* node) {
    EmissionMap emap;
    return ComputePostOrder(node, &emap);
  }
};

}  // namespace ir
}  // namespace torch_xla
