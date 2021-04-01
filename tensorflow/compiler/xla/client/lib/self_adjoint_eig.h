#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace xla {

struct SelfAdjointEigResult {
  XlaOp v;
  XlaOp w;
};

SelfAdjointEigResult SelfAdjointEig(XlaOp a, bool lower = true,
                                    int64 max_iter = 100,
                                    float epsilon = 1e-6) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace xla
