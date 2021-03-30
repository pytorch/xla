#pragma once

#include <cstdint>

#include "lazy_tensors/xla_client/tf_logging.h"

namespace tensorflow {

struct bfloat16 {
  explicit bfloat16(const float v) { TF_LOG(FATAL) << "Not implemented yet."; }

  explicit operator float() const { TF_LOG(FATAL) << "Not implemented yet."; }

  explicit operator unsigned long() const {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  short payload;
};

inline bfloat16 operator-(bfloat16 a, bfloat16 b) {
  TF_LOG(FATAL) << "Not implemented yet.";
}
inline bfloat16 operator/(bfloat16 a, bfloat16 b) {
  TF_LOG(FATAL) << "Not implemented yet.";
}
inline bool operator<(bfloat16 a, bfloat16 b) {
  TF_LOG(FATAL) << "Not implemented yet.";
}
inline bool operator>(bfloat16 a, bfloat16 b) {
  TF_LOG(FATAL) << "Not implemented yet.";
}
inline bfloat16& operator+=(bfloat16& a, bfloat16 b) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace tensorflow
