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

  // Computes the post order from the given node, without using recursion. The
  // emission map can be used as saved state, for multiple separate calls to
  // this API. The returned post-order can be empty if the node has already been
  // emitted inside the emission map. An error is generated if a cyclic loop is
  // detected.
  static std::vector<const Node*> ComputePostOrder(const Node* node,
                                                   EmissionMap* emap);
  static std::vector<const Node*> ComputePostOrder(const Node* node);
};

}  // namespace ir
}  // namespace torch_xla
