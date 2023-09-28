#include "torch_xla/csrc/device.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/types/optional.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/sys_util.h"

namespace torch_xla {
namespace {

thread_local absl::optional<torch::lazy::BackendDevice> g_current_device;

std::string XlaDeviceTypeToString(XlaDeviceType hw_type) {
  switch (hw_type) {
    case XlaDeviceType::CPU:
      return "CPU";
    case XlaDeviceType::GPU:
      return "GPU";
    case XlaDeviceType::TPU:
      return "TPU";
    case XlaDeviceType::XPU:
      return "XPU";
    case XlaDeviceType::NEURON:
      return "NEURON";
    case XlaDeviceType::SPMD:
      return "SPMD";
  }
  XLA_ERROR() << "Invalid device type";
}

// This is set when any device is initialized, so to prevent using non-virtual
// device and virtual device together.
static bool spmd_config_is_locked = false;

}  // namespace

std::string DeviceType::toString() const {
  return absl::StrCat(XlaDeviceTypeToString(static_cast<XlaDeviceType>(type)),
                      ":");
}

torch::lazy::BackendDevice ParseDeviceString(const std::string& device_spec) {
  XLA_CHECK(!device_spec.empty()) << "empty device spec";
  XLA_CHECK(device_spec[0] != ':')
      << "No device type in device specification: " << device_spec;
  std::vector<std::string> device_spec_parts = absl::StrSplit(device_spec, ':');
  XLA_CHECK_EQ(device_spec_parts.size(), 2)
      << "Invalid device specification: " << device_spec;

  int ordinal = std::stoi(device_spec_parts[1]);
  auto device_type = std::make_shared<DeviceType>();
  if (device_spec_parts[0] == "SPMD") {
    device_type->type =
        static_cast<std::underlying_type_t<XlaDeviceType>>(XlaDeviceType::SPMD);
  } else if (device_spec_parts[0] == "TPU") {
    device_type->type =
        static_cast<std::underlying_type_t<XlaDeviceType>>(XlaDeviceType::TPU);
  } else if (device_spec_parts[0] == "CPU") {
    device_type->type =
        static_cast<std::underlying_type_t<XlaDeviceType>>(XlaDeviceType::CPU);
  } else if (device_spec_parts[0] == "GPU") {
    device_type->type =
        static_cast<std::underlying_type_t<XlaDeviceType>>(XlaDeviceType::GPU);
  } else if (device_spec_parts[0] == "XPU") {
    device_type->type =
        static_cast<std::underlying_type_t<XlaDeviceType>>(XlaDeviceType::XPU);
  } else if (device_spec_parts[0] == "NEURON") {
    device_type->type = static_cast<std::underlying_type_t<XlaDeviceType>>(
        XlaDeviceType::NEURON);
  } else {
    XLA_ERROR() << "Invalid device specification: " << device_spec;
  }

  return torch::lazy::BackendDevice(std::move(device_type), ordinal);
}

torch::lazy::BackendDevice GetVirtualDevice() {
  return ParseDeviceString("SPMD:0");
}

bool ShouldUseVirtualDevice() {
  bool use_virtual_device =
      runtime::sys_util::GetEnvBool("XLA_USE_SPMD", false);
  if (use_virtual_device) {
    TF_LOG(INFO) << "Using SPMD virtual device optimization";
  }
  return use_virtual_device;
}

bool UseVirtualDevice() {
  spmd_config_is_locked = true;
  static bool use_virtual_device = ShouldUseVirtualDevice();
  return use_virtual_device;
}

bool GetLockSpmdConfig() { return spmd_config_is_locked; }

}  // namespace torch_xla
