#pragma once

#include <cstddef>
#include <cstdint>

namespace tensorflow {

inline uint64_t Hash64(const char* data, size_t n, uint64_t seed) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace tensorflow
