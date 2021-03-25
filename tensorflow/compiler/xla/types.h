#pragma once

#include <complex>
#include <cstdint>
#include <functional>

#include "tensorflow/core/lib/bfloat16/bfloat16.h"

namespace xla {

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
using bfloat16 = tensorflow::bfloat16;

struct half {
  half() { TF_LOG(FATAL) << "Not implemented yet."; }

  explicit half(const float v) { TF_LOG(FATAL) << "Not implemented yet."; }

  explicit operator float() const { TF_LOG(FATAL) << "Not implemented yet."; }
};

}  // namespace xla
