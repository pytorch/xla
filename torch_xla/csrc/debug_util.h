#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "tensorflow/core/lib/gtl/array_slice.h"
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
      tensorflow::gtl::ArraySlice<const XLATensor> tensors,
      const std::vector<size_t>* indices,
      GraphFormat format = GetDefaultGraphFormat());

  // If the environment variable XLA_SAVE_TENSORS_FILE is set to the proper
  // output path, an instance of the report returned by GetTensorsGraphInfo() is
  // saved.
  static void SaveTensorsGraphInfo(
      const char* name, tensorflow::gtl::ArraySlice<const XLATensor> tensors,
      const std::vector<size_t>* indices,
      GraphFormat format = GetDefaultGraphFormat());
};

}  // namespace torch_xla
