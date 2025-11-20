#include "torch_xla/csrc/device.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <torch/csrc/lazy/backend/backend_device.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/status.h"

namespace torch_xla {
namespace {

// This is set when any device is initialized, so to prevent using non-virtual
// device and virtual device together.
static bool spmd_config_is_locked = false;
static bool use_virtual_device = false;

constexpr int8_t kNativeXlaDeviceTypeNumber =
    static_cast<int8_t>(XlaDeviceType::PLUGIN);

// The elements in this array should match the order in the XlaDeviceType enum
// declaration. So, if you modify one of them, make sure to keep them in sync.
constexpr std::array<std::string_view, kNativeXlaDeviceTypeNumber>
    kNativeXlaDeviceTypeNames = {"CPU", "CUDA", "TPU", "NEURON", "SPMD"};

absl::Status CheckIsNativeXlaDeviceType(int8_t value) {
  if (value < 0 || value >= kNativeXlaDeviceTypeNumber) {
    return XLA_ERROR_WITH_LOCATION(absl::InternalError(
        absl::StrCat("invalid native XlaDeviceType value: ", value,
                     " (casted to int). It should be non-negative, less than ",
                     kNativeXlaDeviceTypeNumber,
                     " (number of native XlaDeviceType). This shouldn't be "
                     "called for XlaDeviceType::PLUGIN (",
                     static_cast<int8_t>(XlaDeviceType::PLUGIN), ").")));
  }
  return absl::OkStatus();
}

std::string_view NativeXlaDeviceTypeToString(XlaDeviceType type) {
  int8_t value = static_cast<int8_t>(type);
  // This check makes sure we are not dealing with:
  //
  //   1. Invalid XlaDeviceType (i.e. result of conversion of a number bigger
  //      than PLUGIN -- the last enum value)
  //
  //   2. The XlaDeviceType::PLUGIN enum, since it's not considered a "native"
  //      device type
  XLA_CHECK_OK(CheckIsNativeXlaDeviceType(value));
  return kNativeXlaDeviceTypeNames[value];
}

XlaDeviceType StringToXlaDeviceType(std::string_view type_name) {
  std::array<std::string_view, kNativeXlaDeviceTypeNumber>::const_iterator it =
      std::find(kNativeXlaDeviceTypeNames.begin(),
                kNativeXlaDeviceTypeNames.end(), type_name);

  if (it == kNativeXlaDeviceTypeNames.end()) {
    return XlaDeviceType::PLUGIN;
  }

  std::size_t index = std::distance(kNativeXlaDeviceTypeNames.begin(), it);
  return static_cast<XlaDeviceType>(index);
}

}  // namespace

DeviceType::DeviceType(XlaDeviceType xla_device_type)
    : torch::lazy::BackendDeviceType(static_cast<int8_t>(xla_device_type)),
      type_name_() {
  XLA_CHECK_OK(CheckIsNativeXlaDeviceType(type));
}

DeviceType::DeviceType(std::string_view type_name)
    : torch::lazy::BackendDeviceType(
          static_cast<int8_t>(StringToXlaDeviceType(type_name))),
      type_name_(type_name) {}

std::string DeviceType::toString() const {
  std::string_view str = (getType() == XlaDeviceType::PLUGIN)
                             ? type_name_
                             : NativeXlaDeviceTypeToString(getType());
  return absl::StrCat(str, ":");
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
