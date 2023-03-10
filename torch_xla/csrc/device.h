#ifndef XLA_TORCH_XLA_CSRC_DEVICE_H_
#define XLA_TORCH_XLA_CSRC_DEVICE_H_

#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/util.h>

#include <iostream>
#include <string>

#include "third_party/xla_client/util.h"

namespace torch_xla {

// TODO(yeounoh) `SPMD` is a virtual device that defers data `TransferToServer`
// until after the paritioning pass. This avoids transfering  the full input
// tensor to the device.
enum class XlaDeviceType { CPU, GPU, TPU, SPMD };

struct DeviceType : public torch::lazy::BackendDeviceType {
  DeviceType() { type = static_cast<int>(XlaDeviceType::CPU); }
  DeviceType(XlaDeviceType xla_device_type) {
    type = static_cast<int>(xla_device_type);
  }

  std::string toString() const override;
};

torch::lazy::BackendDevice ParseDeviceString(const std::string& device_spec);

const torch::lazy::BackendDevice* GetDefaultDevice();

torch::lazy::BackendDevice GetVirtualDevice();

torch::lazy::BackendDevice GetCurrentDevice();

torch::lazy::BackendDevice SetCurrentDevice(
    const torch::lazy::BackendDevice& device);

static inline torch::lazy::BackendDevice GetDeviceOrCurrent(
    const torch::lazy::BackendDevice* device) {
  return device != nullptr ? *device : GetCurrentDevice();
}

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_DEVICE_H_