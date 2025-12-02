#ifndef XLA_TORCH_XLA_CSRC_OPS_DEVICE_DATA_H_
#define XLA_TORCH_XLA_CSRC_OPS_DEVICE_DATA_H_

#include <memory>
#include <string>

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/core/ir.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/runtime/computation_client.h"

namespace torch_xla {

class DeviceData : public XlaNode {
 public:
  DeviceData(torch::lazy::BackendDataPtr data);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  absl::StatusOr<XlaOpVector> SafeLower(LoweringContext* loctx) const override;

  const std::shared_ptr<torch::lazy::BackendData>& data() const {
    return data_;
  }

  void set_buffer_donation(bool should_donate_buffer) {
    std::dynamic_pointer_cast<runtime::ComputationClient::Data>(data_)
        ->set_should_donate_buffer(should_donate_buffer);
  }

  bool get_buffer_donation() {
    return std::dynamic_pointer_cast<runtime::ComputationClient::Data>(data_)
        ->should_donate_buffer();
  }

  static DeviceData* Cast(const torch::lazy::Node* node);

 private:
  // Propagates the sharding stored in `data_` to this node.
  // Specifically, populates `XlaNode::output_shardings_` appropriately.
  absl::Status PropagateShardingFromData();

  torch::lazy::BackendDataPtr data_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_DEVICE_DATA_H_
