#include "torch_xla/csrc/device.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/types/optional.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/sys_util.h"

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
  XLA_CHECK(!device_spec.empty()) << "empty device spec";
  XLA_CHECK(device_spec[0] != ':')
      << "No device type in device specification: " << device_spec;
  std::vector<std::string> device_spec_parts = absl::StrSplit(device_spec, ':');
  XLA_CHECK_EQ(device_spec_parts.size(), 2)
      << "Invalid device specification: " << device_spec;

  int ordinal = std::stoi(device_spec_parts[1]);
  auto device_type = std::make_shared<DeviceType>(device_spec_parts[0]);

  return torch::lazy::BackendDevice(std::move(device_type), ordinal);
}

torch::lazy::BackendDevice GetVirtualDevice() {
  return ParseDeviceString("SPMD:0");
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
