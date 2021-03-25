#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace xla {

inline void QrExplicit(XlaOp a, bool full_matrices, XlaOp& q, XlaOp& r) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace xla
