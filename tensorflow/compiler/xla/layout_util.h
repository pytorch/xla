#pragma once

#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/shape.h"

namespace xla {

class LayoutUtil {
 public:
  static void SetToDefaultLayout(Shape* shape) {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }
};

}  // namespace xla
