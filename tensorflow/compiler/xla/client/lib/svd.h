#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace xla {

struct SVDResult {
  XlaOp u;
  XlaOp d;
  XlaOp v;
};

SVDResult SVD(XlaOp a, int64 max_iter = 100, float epsilon = 1e-6,
              PrecisionConfig::Precision precision = PrecisionConfig::HIGHEST) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace xla
