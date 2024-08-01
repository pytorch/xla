#ifndef XLA_TORCH_XLA_CSRC_UNWRAP_DATA_H_
#define XLA_TORCH_XLA_CSRC_UNWRAP_DATA_H_

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/core/util.h>

#include <vector>

#include "absl/types/span.h"
#include "torch_xla/csrc/runtime/computation_client.h"

namespace torch_xla {

runtime::ComputationClient::DataPtr UnwrapXlaData(
    const torch::lazy::BackendDataPtr& data);

std::vector<runtime::ComputationClient::DataPtr> UnwrapXlaData(
    absl::Span<const torch::lazy::BackendDataPtr> datas);

std::vector<torch::lazy::BackendDataPtr> WrapXlaData(
    absl::Span<const runtime::ComputationClient::DataPtr> xla_datas);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_UNWRAP_DATA_H
