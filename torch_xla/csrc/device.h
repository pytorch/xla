#ifndef XLA_TORCH_XLA_CSRC_DEVICE_H_
#define XLA_TORCH_XLA_CSRC_DEVICE_H_

#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/util.h>

#include <iostream>
#include <string>

#include "torch_xla/csrc/runtime/util.h"

namespace torch_xla {

// TODO(yeounoh) `SPMD` is a virtual device that defers data `TransferToServer`
// until after the paritioning pass. This avoids transfering  the full input
// tensor to the device.
enum class XlaDeviceType { CPU, GPU, TPU, XPU, NEURON, SPMD };

struct DeviceType : public torch::lazy::BackendDeviceType {
  DeviceType() { type = static_cast<int>(XlaDeviceType::CPU); }
  DeviceType(XlaDeviceType xla_device_type) {
    type = static_cast<int>(xla_device_type);
  }

  std::string toString() const override;
};

torch::lazy::BackendDevice ParseDeviceString(const std::string& device_spec);

torch::lazy::BackendDevice GetVirtualDevice();

// Test whether the XLA_USE_SPMD environment variable is set to enable the
// virtual device optimization. This API is called before every device init,
// and sets `spmd_config_is_locked` to block switching the SPMD mode.
bool UseVirtualDevice();

// Return true if SPMD config can be switches. That is, no device has been
// initialized, yet.
bool GetLockSpmdConfig();

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_DEVICE_H_
