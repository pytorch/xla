#ifndef XLA_TORCH_XLA_CSRC_DEBUG_UTIL_H_
#define XLA_TORCH_XLA_CSRC_DEBUG_UTIL_H_

#include <iostream>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {

class DebugUtil {
 public:
  enum GraphFormat {
    kText,
    kDot,
    kHlo,
  };

  static GraphFormat GetDefaultGraphFormat();

  // Dumps the current Python frame and the IR Graph whose roots are the IR
  // values held at the tensors. If indices is not nullptr, it selects the
  // indices of the tensors whose graph will be emitted.
  static std::string GetTensorsGraphInfo(
      absl::Span<const XLATensorPtr> tensors,
      const std::vector<size_t>* indices,
      GraphFormat format = GetDefaultGraphFormat());

  // If the environment variable XLA_SAVE_TENSORS_FILE is set to the proper
  // output path, an instance of the report returned by GetTensorsGraphInfo() is
  // saved.
  static void SaveTensorsGraphInfo(
      const char* name, absl::Span<const XLATensorPtr> tensors,
      const std::vector<size_t>* indices,
      GraphFormat format = GetDefaultGraphFormat());

  static bool ExperimentEnabled(const std::string& name);
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_DEBUG_UTIL_H_