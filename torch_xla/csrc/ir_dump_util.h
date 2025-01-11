#ifndef XLA_TORCH_XLA_CSRC_IR_DUMP_UTIL_H_
#define XLA_TORCH_XLA_CSRC_IR_DUMP_UTIL_H_

#include <string>

#include "absl/types/span.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {

enum class EmitMode {
  kHloReadable,
  kHloProto,
  kStableHloReadable,
  kStableHloBytecode,
};

class DumpUtil {
 public:
  static std::string ToDot(absl::Span<const torch::lazy::Node* const> nodes);

  static std::string PostOrderToDot(
      absl::Span<const torch::lazy::Node* const> post_order,
      absl::Span<const torch::lazy::Node* const> roots);

  static std::string ToText(absl::Span<const torch::lazy::Node* const> nodes);

  static std::string PostOrderToText(
      absl::Span<const torch::lazy::Node* const> post_order,
      absl::Span<const torch::lazy::Node* const> roots);

  static std::string ToHlo(c10::ArrayRef<torch::lazy::Value> values,
                           const torch::lazy::BackendDevice& device,
                           EmitMode mode = EmitMode::kHloReadable);
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_IR_DUMP_UTIL_H_