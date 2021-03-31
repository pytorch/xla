#pragma once

#include "lazy_tensors/computation_client/debug_macros.h"
#include "tensorflow/compiler/xla/statusor.h"

template <typename T>
T ConsumeValue(xla::StatusOr<T>&& status) {
  LTC_CHECK_OK(status.status());
  return status.ConsumeValueOrDie();
}
