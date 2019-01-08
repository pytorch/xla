#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch_xla {

// Recognizes a gt, type_as, mul sequence and replaces it with
// threshold_backward. Works around an issue in the TPU compiler with S8 tensor
// shapes.
void ThresholdBackwardPeephole(const std::shared_ptr<torch::jit::Graph>& graph);

}  // namespace torch_xla
