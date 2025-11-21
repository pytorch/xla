#include "torch_xla/csrc/ops/device_data.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/core/ir.h>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/xla_data.pb.h"

#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/runtime.h"
#include "torch_xla/csrc/status.h"

namespace torch_xla {

DeviceData::DeviceData(torch::lazy::BackendDataPtr data)
    : XlaNode(xla_device_data,
              std::dynamic_pointer_cast<runtime::ComputationClient::Data>(data)
                  ->shape(),
              /*num_outputs=*/1,
              /*hash_seed=*/(uint32_t)101),
      data_(std::move(data)) {
  XLA_THROW_IF_ERROR(PropagateShardingFromData());
}

std::string DeviceData::ToString() const {
  return absl::StrCat(XlaNode::ToString(),
                      ", device=", data_->device().toString());
}

torch::lazy::NodePtr DeviceData::Clone(torch::lazy::OpList operands) const {
  return torch_xla::MakeNode<DeviceData>(data_);
}

absl::StatusOr<XlaOpVector> DeviceData::SafeLower(
    LoweringContext* loctx) const {
  XLA_ASSIGN_OR_RETURN(xla::XlaOp op,
                       loctx->GetParameter(data_, unbounded_dynamic_dims_));
  return ReturnOp(op, loctx);
}

DeviceData* DeviceData::Cast(const torch::lazy::Node* node) {
  return torch_xla::NodeCast<DeviceData>(node, xla_device_data);
}

absl::Status DeviceData::PropagateShardingFromData() {
  XLA_ASSIGN_OR_RETURN(runtime::ComputationClient * absl_nonnull const client,
                       runtime::GetComputationClient());
  XLA_ASSIGN_OR_RETURN(absl_nonnull runtime::ComputationClient::DataPtr cc_data,
                       runtime::AsComputationClientData(data_));

  std::optional<xla::OpSharding> op_sharding = client->GetDataSharding(cc_data);
  if (op_sharding.has_value()) {
    // DeviceData Node only has 1 output.
    SetSharding(op_sharding.value(), 0);
  }

  return absl::OkStatus();
}

}  // namespace torch_xla
