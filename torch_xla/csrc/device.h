#pragma once

#include <iostream>
#include <string>

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/lazy/backend/backend_device.h"
#include "torch/csrc/lazy/core/hash.h"
#include "torch/csrc/lazy/core/util.h"

namespace torch_xla {

enum class TorchXLADeviceType { CPU, GPU, TPU };

struct DeviceType : public torch::lazy::BackendDeviceType {
  TorchXLADeviceType hw_type;

  DeviceType(TorchXLADeviceType torch_xla_device_type) {
    hw_type = torch_xla_device_type;
    type = static_cast<int>(hw_type);
  }
};

struct Device : public torch::lazy::BackendDevice {
  Device() = default;
  explicit Device(const std::string& device_spec);
  Device(DeviceType device_type, int ordinal)
      : torch::lazy::BackendDevice(
            std::make_shared<torch::lazy::BackendDeviceType>(device_type),
            ordinal),
        device_type(device_type),
        ordinal(ordinal) {}

  bool operator==(const Device& other) const { return compare(other) == 0; }

  bool operator!=(const Device& other) const { return compare(other) != 0; }

  bool operator<(const Device& rhs) const { return compare(rhs) < 0; }

  int compare(const Device& rhs) const {
    if (device_type.hw_type != rhs.device_type.hw_type) {
      return device_type.hw_type < rhs.device_type.hw_type ? -1 : +1;
    }
    return ordinal < rhs.ordinal ? -1 : (ordinal > rhs.ordinal ? +1 : 0);
  }

  std::string ToString() const;

  size_t hash() const {
    return torch::lazy::StdHashCombine(
        torch::lazy::GetEnumValue(device_type.hw_type), ordinal + 1);
  }

  DeviceType device_type = DeviceType(TorchXLADeviceType::CPU);
  int ordinal = 0;
};

const Device* GetDefaultDevice();

Device GetCurrentDevice();

Device SetCurrentDevice(const Device& device);

static inline Device GetDeviceOrCurrent(const Device* device) {
  return device != nullptr ? *device : GetCurrentDevice();
}

}  // namespace torch_xla
