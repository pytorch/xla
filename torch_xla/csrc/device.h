#ifndef XLA_TORCH_XLA_CSRC_DEVICE_H_
#define XLA_TORCH_XLA_CSRC_DEVICE_H_

#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/util.h>

#include <iostream>
#include <string>

#include "torch_xla/csrc/runtime/util.h"

namespace torch_xla {

// TODO(yeounoh) `SPMD` is a virtual device that defers data `TransferToDevice`
// until after the paritioning pass. This avoids transfering  the full input
// tensor to the device.
enum class XlaDeviceType { CPU, CUDA, TPU, NEURON, SPMD, PLUGIN };

struct DeviceType : public torch::lazy::BackendDeviceType {
  DeviceType(XlaDeviceType xla_device_type)
      : torch::lazy::BackendDeviceType(static_cast<int>(xla_device_type)),
        type_name_(XlaDeviceTypeToString(xla_device_type)) {}
  DeviceType(const std::string& type_name)
      : torch::lazy::BackendDeviceType(
            static_cast<int>(StringToXlaDeviceType(type_name))),
        type_name_(type_name) {}

  std::string toString() const override;
  XlaDeviceType getType() const;

 private:
  std::string type_name_;

  static std::string XlaDeviceTypeToString(XlaDeviceType hw_type);
  static XlaDeviceType StringToXlaDeviceType(const std::string& type_name);
};

torch::lazy::BackendDevice ParseDeviceString(const std::string& device_spec);

torch::lazy::BackendDevice GetVirtualDevice();

// Test whether the XLA_USE_SPMD environment variable is set to enable the
// virtual device optimization. This API is called before every device init,
// and sets `spmd_config_is_locked` to block switching the SPMD mode.
// Optionally, `force_spmd` to set `use_virtual_device` to true.
bool UseVirtualDevice(bool force_spmd = false);

// Return true if `device` is of SPMD device type.
bool IsVirtualDevice(const std::string& device);

// Return true if SPMD config can be switches. That is, no device has been
// initialized, yet.
bool GetLockSpmdConfig();

// Return true if the physical device type is TPU.
// TODO(yeounoh) - see if we need to check for AOT compilation device type.
bool CheckTpuDevice(XlaDeviceType hw_type);

// Return true if the physical device type is NEURON.
bool CheckNeuronDevice(XlaDeviceType hw_type);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_DEVICE_H_
