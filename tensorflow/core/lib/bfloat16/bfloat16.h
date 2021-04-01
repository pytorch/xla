#pragma once

#include <cstdint>

#include "lazy_tensors/computation_client/ltc_logging.h"

namespace tensorflow {

struct bfloat16 {
  explicit bfloat16(const float v) { LTC_LOG(FATAL) << "Not implemented yet."; }

  explicit operator float() const { LTC_LOG(FATAL) << "Not implemented yet."; }

  explicit operator unsigned long() const {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  short payload;
};

inline bfloat16 operator-(bfloat16 a, bfloat16 b) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}
inline bfloat16 operator/(bfloat16 a, bfloat16 b) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}
inline bool operator<(bfloat16 a, bfloat16 b) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}
inline bool operator>(bfloat16 a, bfloat16 b) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}
inline bfloat16& operator+=(bfloat16& a, bfloat16 b) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace tensorflow
