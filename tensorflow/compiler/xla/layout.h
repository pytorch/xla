#pragma once

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class Tile {};

class Layout {
 public:
  absl::Span<const int64> minor_to_major() const {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }
};

}  // namespace xla
