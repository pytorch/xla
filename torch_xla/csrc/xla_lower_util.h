#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

xla::XlaOp CreateMatMul(const xla::XlaOp& lhs, const xla::XlaOp& rhs);

}  // namespace torch_xla
