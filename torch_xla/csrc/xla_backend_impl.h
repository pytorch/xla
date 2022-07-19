#pragma once

#include <iostream>
#include <string>

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "torch/csrc/lazy/backend/backend_interface.h"
#include "torch_xla/csrc/device.h"

namespace torch_xla {
class XLAData : public torch::lazy::BackendData {
 public:
  XLAData(const torch::lazy::Shape& shape,
          const torch::lazy::BackendDevice& device,
          xla::ComputationClient::DataPtr xla_data)
      : torch::lazy::BackendData(device, shape), xla_data_(xla_data) {}

  // TODO set Device and torch::lazy_shape correctly
  XLAData(xla::ComputationClient::DataPtr xla_data)
      : torch::lazy::BackendData(ParseDeviceString(xla_data->device()),
                                 torch::lazy::Shape()),
        xla_data_(xla_data) {}

  Handle GetHandle() override { return xla_data_->GetOpaqueHandle(); }

  void Assign(const torch::lazy::BackendData& data) override {
    // Assign should only be called to update the handle, no need
    // to update device and shape.
    XLAData new_xla_data = static_cast<const XLAData&>(data).xla_data_;
    xla_data_->Assign(*(new_xla_data.xla_data().get()));
  }

  bool HasValue() const override { return xla_data_->HasValue(); }

  xla::ComputationClient::DataPtr xla_data() { return xla_data_; }

 private:
  // TODO: Do we really need a Share_Ptr here?
  xla::ComputationClient::DataPtr xla_data_;
};

torch::lazy::BackendImplInterface* GetXlaBackendImpl();

void InitXlaBackend();

}  // namespace torch_xla
