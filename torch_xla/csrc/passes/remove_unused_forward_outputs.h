#pragma once

#include "torch/csrc/jit/autodiff.h"

namespace torch {
namespace jit {

// Remove outputs from forward graph which are not useful for the XLA lowering.
void RemoveUnusedForwardOutputs(Gradient* gradient);

}  // namespace jit
}  // namespace torch
