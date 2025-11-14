#include "torch_xla/csrc/device.h"

#include <memory>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/status.h"

namespace torch_xla {
namespace {

// This is set when any device is initialized, so to prevent using non-virtual
// device and virtual device together.
static bool spmd_config_is_locked = false;
static bool use_virtual_device = false;

}  // namespace

std::string DeviceType::XlaDeviceTypeToString(XlaDeviceType hw_type) {
  XLA_CHECK(hw_type != XlaDeviceType::PLUGIN) << "PLUGIN type name unknown";

  switch (hw_type) {
    case XlaDeviceType::CPU:
      return "CPU";
    case XlaDeviceType::CUDA:
      return "CUDA";
    case XlaDeviceType::TPU:
      return "TPU";
    case XlaDeviceType::NEURON:
      return "NEURON";
    case XlaDeviceType::SPMD:
      return "SPMD";
    default:
      XLA_ERROR() << "Invalid device type";
  }
}

XlaDeviceType DeviceType::StringToXlaDeviceType(const std::string& type_name) {
  if (type_name == "SPMD") {
    return XlaDeviceType::SPMD;
  } else if (type_name == "TPU") {
    return XlaDeviceType::TPU;
  } else if (type_name == "CPU") {
    return XlaDeviceType::CPU;
  } else if (type_name == "CUDA") {
    return XlaDeviceType::CUDA;
  } else if (type_name == "NEURON") {
    return XlaDeviceType::NEURON;
  }

  return XlaDeviceType::PLUGIN;
}

std::string DeviceType::toString() const {
  return absl::StrCat(type_name_, ":");
}

XlaDeviceType DeviceType::getType() const {
  return static_cast<XlaDeviceType>(type);
}

torch::lazy::BackendDevice ParseDeviceString(const std::string& device_spec) {
  XLA_ASSIGN_OR_THROW(torch::lazy::BackendDevice device,
                      SafeParseDeviceString(device_spec));
  return device;
}

absl::StatusOr<torch::lazy::BackendDevice> SafeParseDeviceString(
    const std::string& device_spec) {
  std::vector<std::string> parts = absl::StrSplit(device_spec, ':');

  if (parts.size() != 2) {
    return XLA_ERROR_WITH_LOCATION(absl::InvalidArgumentError(
        absl::StrCat("expected the device string `", device_spec,
                     "` to be in the format: `<type>:<index>`.")));
  }

  const std::string& type_str = parts[0];
  const std::string& index_str = parts[1];

  try {
    return torch::lazy::BackendDevice(std::make_shared<DeviceType>(type_str),
                                      std::stoi(index_str));
  } catch (const std::exception& e) {
    return XLA_ERROR_WITH_LOCATION(absl::InvalidArgumentError(
        absl::StrCat("error while parsing the device spec `", device_spec,
                     "`: ", e.what())));
  }
}

torch::lazy::BackendDevice GetVirtualDevice() {
  return torch::lazy::BackendDevice(
      std::make_shared<DeviceType>(XlaDeviceType::SPMD), 0);
}

bool ShouldUseVirtualDevice() {
  bool use_virtual_device =
      runtime::sys_util::GetEnvBool("XLA_USE_SPMD", false) ||
      runtime::sys_util::GetEnvBool("XLA_AUTO_SPMD", false);
  return use_virtual_device;
}

bool UseVirtualDevice(bool force_spmd) {
  spmd_config_is_locked = true;
  use_virtual_device = ShouldUseVirtualDevice();
  if (force_spmd) {
    use_virtual_device = true;
  }
  return use_virtual_device;
}

bool IsVirtualDevice(const std::string& device) {
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(ParseDeviceString(device).type());
  return hw_type == XlaDeviceType::SPMD;
}

bool GetLockSpmdConfig() { return spmd_config_is_locked; }

bool CheckTpuDevice(XlaDeviceType hw_type) {
  if (hw_type == XlaDeviceType::TPU) {
    return true;
  }

  std::string pjrt_device = runtime::sys_util::GetEnvString("PJRT_DEVICE", "");
  if (hw_type == XlaDeviceType::SPMD) {
    return pjrt_device == "TPU";
  }
  return false;
}

bool CheckNeuronDevice(XlaDeviceType hw_type) {
  if (hw_type == XlaDeviceType::NEURON) {
    return true;
  }

  std::string pjrt_device = runtime::sys_util::GetEnvString("PJRT_DEVICE", "");
  if (hw_type == XlaDeviceType::SPMD) {
    return pjrt_device == "NEURON";
  }
  return false;
}

}  // namespace torch_xla
