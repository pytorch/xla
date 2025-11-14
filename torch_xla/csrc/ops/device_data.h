#ifndef XLA_TORCH_XLA_CSRC_OPS_DEVICE_DATA_H_
#define XLA_TORCH_XLA_CSRC_OPS_DEVICE_DATA_H_

#include <torch/csrc/lazy/backend/backend_data.h>

#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/runtime/computation_client.h"

namespace torch_xla {

class DeviceData : public XlaNode {
 public:
  DeviceData(std::shared_ptr<torch::lazy::BackendData> data);

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

  // With SPMD sharding propagation, we need to update the unpartitioned
  // backend data with a partitioned one in the node operands. Note that
  // this is permitted only if the node holds a placeholder.
  void Assign(std::shared_ptr<torch::lazy::BackendData> data) {
    XLA_CHECK(data->shape() == data_->shape())
        << "Shape mismatch: expected (" << data_->shape().to_string()
        << "), actual (" << data->shape().to_string() << ")";
    data_.reset(data.get());
  }

  static DeviceData* Cast(const torch::lazy::Node* node);

 private:
  std::shared_ptr<torch::lazy::BackendData> data_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_DEVICE_DATA_H_
