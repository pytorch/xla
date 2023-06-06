#include "torch_xla/csrc/xla_data.h"

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/shape.h>

#include "third_party/xla_client/computation_client.h"
#include "torch/csrc/lazy/backend/backend_data.h"
#include "torch_xla/csrc/device.h"

namespace torch_xla {

XLAData::XLAData(const torch::lazy::Shape& shape,
                 const torch::lazy::BackendDevice& device,
                 xla::ComputationClient::DataPtr xla_data)
    : torch::lazy::BackendData(device, shape), xla_data_(xla_data) {}

// TODO set Device and torch::lazy_shape correctly
XLAData::XLAData(xla::ComputationClient::DataPtr xla_data)
    : torch::lazy::BackendData(ParseDeviceString(xla_data->device()),
                               torch::lazy::Shape()),
      xla_data_(xla_data) {}

torch::lazy::BackendData::Handle XLAData::GetHandle() {
  return xla_data_->GetOpaqueHandle();
}

void XLAData::Assign(const torch::lazy::BackendData& data) {
  // Assign should only be called to update the handle, no need
  // to update device and shape.
  XLAData new_xla_data = static_cast<const XLAData&>(data).xla_data_;
  xla_data_->Assign(*(new_xla_data.xla_data().get()));
}

bool XLAData::HasValue() const { return xla_data_->HasValue(); }

xla::ComputationClient::DataPtr XLAData::xla_data() { return xla_data_; }

}  // namespace torch_xla
