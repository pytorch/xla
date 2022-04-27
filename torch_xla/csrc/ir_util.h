#pragma once

#include <unordered_map>
#include <vector>

#include "absl/types/span.h"
#include "torch/csrc/lazy/core/ir_util.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {

class Util {
 public:
  using EmissionMap = std::unordered_map<const torch::lazy::Node*,
                                         torch::lazy::Util::EmitStatus>;

  // Computes the post order from the given node, without using recursion. The
  // emission map can be used as saved state, for multiple separate calls to
  // this API. The returned post-order can be empty if the node has already been
  // emitted inside the emission map. An error is generated if a loop is
  // detected.
  static std::vector<const torch::lazy::Node*> ComputePostOrder(
      const torch::lazy::Node* node, EmissionMap* emap);

  static std::vector<const torch::lazy::Node*> ComputePostOrder(
      absl::Span<const torch::lazy::Node* const> nodes, EmissionMap* emap);

  // Same as above, but computes the post order on the set of nodes specified as
  // argument.
  static std::vector<const torch::lazy::Node*> ComputePostOrder(
      absl::Span<const torch::lazy::Node* const> nodes);

  // Clones the IR graph whose roots are passed in the values parameter.
  static std::vector<XlaValue> Clone(absl::Span<const XlaValue> values);

  // Same as the above, but the post-order is passed as parameter.
  static std::vector<XlaValue> Clone(
      absl::Span<const XlaValue> values,
      absl::Span<const torch::lazy::Node* const> post_order);

  // Retrieves the number of nodes within the graph whose sink are passed in the
  // nodes argument.
  static size_t GetGraphSize(absl::Span<const torch::lazy::Node* const> nodes);
};

}  // namespace ir
}  // namespace torch_xla
