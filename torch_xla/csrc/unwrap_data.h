#ifndef XLA_TORCH_XLA_CSRC_UNWRAP_DATA_H_
#define XLA_TORCH_XLA_CSRC_UNWRAP_DATA_H_

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/core/util.h>

#include <vector>

#include "absl/types/span.h"
#include "third_party/xla_client/computation_client.h"

namespace torch_xla {
xla::ComputationClient::DataPtr UnwrapXlaData(
    const torch::lazy::BackendDataPtr& data);

std::vector<xla::ComputationClient::DataPtr> UnwrapXlaData(
    absl::Span<const torch::lazy::BackendDataPtr> datas);

torch::lazy::BackendDataPtr WrapXlaData(
    const xla::ComputationClient::DataPtr& xla_data);

std::vector<torch::lazy::BackendDataPtr> WrapXlaData(
    absl::Span<const xla::ComputationClient::DataPtr> xla_datas);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_UNWRAP_DATA_H
