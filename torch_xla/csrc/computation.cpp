#include "torch_xla/csrc/computation.h"

#include "torch_xla/csrc/runtime/debug_macros.h"

namespace torch_xla {

std::vector<torch::lazy::ComputationPtr> WrapClientComputation(
    std::vector<std::shared_ptr<runtime::ComputationClient::Computation>>
        computations) {
  std::vector<torch::lazy::ComputationPtr> res;
  res.reserve(computations.size());
  for (auto client_computation : computations) {
    res.push_back(std::dynamic_pointer_cast<torch::lazy::Computation>(client_computation));
  }
  return res;
}

std::shared_ptr<runtime::ComputationClient::Computation>
UnwrapClientComputation(torch::lazy::ComputationPtr computation) {
  return std::dynamic_pointer_cast<runtime::ComputationClient::Computation>(computation);
}

}  // namespace torch_xla
