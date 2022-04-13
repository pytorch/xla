#pragma once

#include <iostream>
#include <string>

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/lazy/backend/backend_device.h"
#include "torch/csrc/lazy/core/hash.h"
#include "torch/csrc/lazy/core/util.h"

namespace torch_xla {

enum class XlaDeviceType { CPU, GPU, TPU };

struct DeviceType : public torch::lazy::BackendDeviceType {
  DeviceType() { type = static_cast<int>(XlaDeviceType::CPU); }
  DeviceType(XlaDeviceType xla_device_type) {
    type = static_cast<int>(xla_device_type);
  }

  std::string toString() const;
};

// struct Device : public torch::lazy::BackendDevice {
//   Device() = default;
//   explicit Device(const std::string& device_spec);
//   Device(DeviceType device_type, int ordinal)
//       : torch::lazy::BackendDevice(
//             std::make_shared<torch::lazy::BackendDeviceType>(device_type),
//             ordinal),
//         device_type(device_type),
//         ordinal(ordinal) {}

//   bool operator==(const Device& other) const { return compare(other) == 0; }

//   bool operator!=(const Device& other) const { return compare(other) != 0; }

//   bool operator<(const Device& rhs) const { return compare(rhs) < 0; }

//   int compare(const Device& rhs) const {
//     if (device_type.hw_type != rhs.type()) {
//       return device_type.hw_type < rhs.type() ? -1 : +1;
//     }
//     return ordinal < rhs.ordinal ? -1 : (ordinal > rhs.ordinal ? +1 : 0);
//   }

//   std::string ToString() const;

//   size_t hash() const {
//     return torch::lazy::StdHashCombine(
//         torch::lazy::GetEnumValue(device_type.hw_type), ordinal + 1);
//   }

//   DeviceType device_type = DeviceType(XlaDeviceType::CPU);
//   int ordinal = 0;
// };

torch::lazy::BackendDevice ParseDeviceString(const std::string& device_spec);

const torch::lazy::BackendDevice* GetDefaultDevice();

torch::lazy::BackendDevice GetCurrentDevice();

torch::lazy::BackendDevice SetCurrentDevice(
    const torch::lazy::BackendDevice& device);

static inline torch::lazy::BackendDevice GetDeviceOrCurrent(
    const torch::lazy::BackendDevice* device) {
  return device != nullptr ? *device : GetCurrentDevice();
}

}  // namespace torch_xla
