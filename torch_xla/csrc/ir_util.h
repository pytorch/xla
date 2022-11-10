#pragma once

#include <unordered_map>
#include <vector>

#include "absl/types/span.h"
#include "torch/csrc/lazy/core/ir_util.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Util {
 public:
  // Clones the IR graph whose roots are passed in the values parameter.
  static std::vector<torch::lazy::Value> Clone(
      c10::ArrayRef<torch::lazy::Value> values);

  // Same as the above, but the post-order is passed as parameter.
  static std::vector<torch::lazy::Value> Clone(
      c10::ArrayRef<torch::lazy::Value> values,
      absl::Span<const torch::lazy::Node* const> post_order);
};

}  // namespace torch_xla
