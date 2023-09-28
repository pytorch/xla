#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/util.h>

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "torch_xla/csrc/runtime/computation_client.h"

namespace torch_xla {

runtime::ComputationClient::DataPtr UnwrapXlaData(
    const torch::lazy::BackendDataPtr& data) {
  TORCH_LAZY_TIMED("UnwrapXlaData");
  return std::dynamic_pointer_cast<runtime::ComputationClient::Data>(data);
}

std::vector<runtime::ComputationClient::DataPtr> UnwrapXlaData(
    absl::Span<const torch::lazy::BackendDataPtr> datas) {
  TORCH_LAZY_TIMED("UnwrapXlaData");
  std::vector<runtime::ComputationClient::DataPtr> xla_datas;
  xla_datas.reserve(datas.size());
  for (const auto& data : datas) {
    xla_datas.push_back(
        std::dynamic_pointer_cast<runtime::ComputationClient::Data>(data));
  }
  return xla_datas;
}

std::vector<torch::lazy::BackendDataPtr> WrapXlaData(
    absl::Span<const runtime::ComputationClient::DataPtr> xla_datas) {
  TORCH_LAZY_TIMED("WrapXlaData");
  return {xla_datas.begin(), xla_datas.end()};
}

}  // namespace torch_xla
