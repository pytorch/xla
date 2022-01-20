#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "torch/csrc/lazy/core/ir_metadata.h"

namespace torch_xla {

absl::optional<torch::lazy::SourceLocation> GetPythonFrameTop();

std::vector<torch::lazy::SourceLocation> GetPythonFrames();

}  // namespace torch_xla
