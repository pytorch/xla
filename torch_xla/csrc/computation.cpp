#include "torch_xla/csrc/computation.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {

Computation::Computation(std::string name, xla::XlaComputation computation)
    : name_(std::move(name)),
      xla_client_computation_(
          std::make_shared<xla::ComputationClient::Computation>(
              std::move(computation))) {
  hash_ = torch::lazy::MHash(
      name_,
      xla_client_computation_->computation().proto().SerializeAsString());
}

Computation::Computation(
    std::shared_ptr<xla::ComputationClient::Computation> xla_client_computation)
    : name_(""), hash_(0) {
  xla_client_computation_ = xla_client_computation;
}

std::vector<torch::lazy::ComputationPtr> WrapClientComputation(
    std::vector<std::shared_ptr<xla::ComputationClient::Computation>>
        computations) {
  std::vector<torch::lazy::ComputationPtr> res;
  res.reserve(computations.size());
  for (auto client_computation : computations) {
    res.push_back(std::make_shared<Computation>(client_computation));
  }
  return res;
}

}  // namespace torch_xla
