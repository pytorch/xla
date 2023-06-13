#include <ATen/Functions.h>
#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/util.h>

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/xla_data.h"

namespace torch_xla {

runtime::ComputationClient::DataPtr UnwrapXlaData(
    const torch::lazy::BackendDataPtr& data) {
  TORCH_LAZY_TIMED("UnwrapXlaData");
  return dynamic_cast<XLAData*>(data.get())->xla_data();
}

std::vector<runtime::ComputationClient::DataPtr> UnwrapXlaData(
    absl::Span<const torch::lazy::BackendDataPtr> datas) {
  TORCH_LAZY_TIMED("UnwrapXlaData");
  std::vector<runtime::ComputationClient::DataPtr> xla_datas;
  xla_datas.reserve(datas.size());
  for (const auto& data : datas) {
    xla_datas.push_back(dynamic_cast<XLAData*>(data.get())->xla_data());
  }
  return xla_datas;
}

torch::lazy::BackendDataPtr WrapXlaData(
    const runtime::ComputationClient::DataPtr& xla_data) {
  TORCH_LAZY_TIMED("WrapXlaData");
  return std::make_shared<XLAData>(xla_data);
}

std::vector<torch::lazy::BackendDataPtr> WrapXlaData(
    absl::Span<const runtime::ComputationClient::DataPtr> xla_datas) {
  TORCH_LAZY_TIMED("WrapXlaData");
  std::vector<torch::lazy::BackendDataPtr> datas;
  datas.reserve(xla_datas.size());
  for (const auto& xla_data : xla_datas) {
    datas.push_back(std::make_shared<XLAData>(xla_data));
  }
  return datas;
}

}  // namespace torch_xla
