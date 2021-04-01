#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace xla {

inline XlaOp TorchGather(XlaOp input, XlaOp index, int64 dim,
                         bool sparse = true) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp TorchIndexSelect(XlaOp input, XlaOp index, int64 dim,
                              int64 batch_dims = 0) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace xla
