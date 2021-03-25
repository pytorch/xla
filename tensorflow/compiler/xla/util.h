#pragma once

#include "absl/types/span.h"

namespace xla {

inline absl::Span<const int64> AsInt64Slice(absl::Span<const int64> slice) {
  return slice;
}

}  // namespace xla
