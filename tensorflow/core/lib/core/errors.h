#pragma once

#include "tensorflow/compiler/xla/status.h"

namespace tensorflow {
namespace errors {

inline xla::Status Internal(const char* message) {
  return xla::Status(message);
}

}  // namespace errors
}  // namespace tensorflow
