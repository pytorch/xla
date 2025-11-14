#ifndef XLA_TORCH_XLA_CSRC_DEVICE_H_
#define XLA_TORCH_XLA_CSRC_DEVICE_H_

#include <string>
#include <string_view>

#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/util.h>

#include "absl/status/statusor.h"

namespace torch_xla {

// Convenient macro for applying another macro to all native device types.
//
// Add new device type
// ===================
//
// Add a new line to the macro below:
//
//     _(<DEVICE>, <INDEX>)
//
// Where <DEVICE> is the enum of the given device, and <INDEX> is the
// previous number plus 1.
//
#define XLA_FOR_ALL_NATIVE_DEVICE_TYPES_(_) \
  _(CPU, 0)                                 \
  _(CUDA, 1)                                \
  _(TPU, 2)                                 \
  _(NEURON, 3)                              \
  _(SPMD, 4)

// TODO(yeounoh) `SPMD` is a virtual device that defers data `TransferToDevice`
// until after the paritioning pass. This avoids transfering  the full input
// tensor to the device.
enum class XlaDeviceType : int8_t {
#define XLA_DECLARE_ENUM(name, value) name = value,
  XLA_FOR_ALL_NATIVE_DEVICE_TYPES_(XLA_DECLARE_ENUM)
#undef XLA_DECLARE_ENUM

  // Plugin is not considered a native device type.
  // It has a special treatment for some functions.
  PLUGIN,
};

struct DeviceType : public torch::lazy::BackendDeviceType {
  DeviceType(XlaDeviceType xla_device_type);

  // Constructor parses the `type_name` into an `XlaDeviceType`.
  //
  // This should be used in 2 cases:
  //
  //   1. When using non-native device types.
  //      Although `XlaDeviceType::PLUGIN` will be used, the `type_name`
  //      parameter will be stored internally.
  //
  //   2. When parsing string device types.
  DeviceType(std::string_view type_name);

  std::string toString() const override;
  XlaDeviceType getType() const;

 private:
  std::string type_name_;
};

// Parses the given `device_spec` into a new `BackendDevice`.
//
// Prefer its safer version (i.e. SafeParseDeviceString), since this function
// throws an exception on error, instead of returning a status instance.
[[deprecated("Use SafeParseDeviceString for better error handling.")]] torch::
    lazy::BackendDevice
    ParseDeviceString(const std::string& device_spec);

// Parses the given `device_spec` into a new `BackendDevice`.
//
// This function returns an error status if:
//   1. `device_spec` is not in the format: `<type>:<index>`
//   2. `<type>` is not any of `XlaDeviceType`
//   3. `<index>` is not an integer
absl::StatusOr<torch::lazy::BackendDevice> SafeParseDeviceString(
    const std::string& device_spec);

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
