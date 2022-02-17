#include "torch_xla/csrc/device.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace {

thread_local absl::optional<Device> g_current_device;

std::string DeviceTypeToString(DeviceType device_type) {
  switch (device_type.hw_type) {
    case TorchXLADeviceType::CPU:
      return "CPU";
    case TorchXLADeviceType::GPU:
      return "GPU";
    case TorchXLADeviceType::TPU:
      return "TPU";
  }
  XLA_ERROR() << "Invalid device type";
}

void ParseDevice(const std::string& device_spec, Device* device) {
  if (device_spec.empty()) {
    std::string default_device_spec =
        xla::ComputationClient::Get()->GetDefaultDevice();
    XLA_CHECK(!default_device_spec.empty());
    return ParseDevice(default_device_spec, device);
  }
  if (device_spec[0] == ':') {
    std::string default_device_spec =
        xla::ComputationClient::Get()->GetDefaultDevice();
    auto pos = default_device_spec.find(':');
    XLA_CHECK_NE(pos, std::string::npos) << default_device_spec;
    return ParseDevice(default_device_spec.substr(0, pos) + device_spec,
                       device);
  }
  std::vector<std::string> device_spec_parts = absl::StrSplit(device_spec, ':');
  XLA_CHECK_EQ(device_spec_parts.size(), 2)
      << "Invalid device specification: " << device_spec;

  device->ordinal = std::stoi(device_spec_parts[1]);
  if (device_spec_parts[0] == "TPU") {
    device->device_type.hw_type = TorchXLADeviceType::TPU;
  } else if (device_spec_parts[0] == "CPU") {
    device->device_type.hw_type = TorchXLADeviceType::CPU;
  } else if (device_spec_parts[0] == "GPU") {
    device->device_type.hw_type = TorchXLADeviceType::GPU;
  } else {
    XLA_ERROR() << "Invalid device specification: " << device_spec;
  }
}

}  // namespace

Device::Device(const std::string& device_spec) {
  ParseDevice(device_spec, this);
}

std::string Device::ToString() const {
  return absl::StrCat(DeviceTypeToString(device_type.hw_type), ":", ordinal);
}

const Device* GetDefaultDevice() {
  static const Device* default_device = new Device("");
  return default_device;
}

Device GetCurrentDevice() {
  if (!g_current_device) {
    g_current_device = *GetDefaultDevice();
  }
  return *g_current_device;
}

Device SetCurrentDevice(const Device& device) {
  Device current = GetCurrentDevice();
  g_current_device = device;
  TF_VLOG(2) << "New current device: " << device;
  return current;
}

}  // namespace torch_xla
