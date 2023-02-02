#pragma once

#include "third_party/xla_client/computation_client.h"
#include "torch/csrc/lazy/backend/backend_data.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class DeviceData : public XlaNode {
 public:
  DeviceData(std::shared_ptr<torch::lazy::BackendData> data);

  DeviceData(std::shared_ptr<torch::lazy::BackendData> data, torch::lazy::OpList ops, xla::OpSharding sharding);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  torch::lazy::NodePtr CloneWithSharding(xla::OpSharding sharding) const;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::shared_ptr<torch::lazy::BackendData>& data() const {
    return data_;
  }

  static DeviceData* Cast(const torch::lazy::Node* node);

 private:
  std::shared_ptr<torch::lazy::BackendData> data_;
};

}  // namespace torch_xla
