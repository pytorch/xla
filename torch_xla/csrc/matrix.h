#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch/csrc/jit/ir.h"

namespace torch_xla {

xla::XlaOp BuildTriu(const xla::XlaOp& input, int diagonal);

xla::XlaOp BuildTril(const xla::XlaOp& input, int diagonal);

}  // namespace torch_xla
