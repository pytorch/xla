#pragma once

#include <string>

// PyTorch integration headers
#include <ATen/core/Generator.h>
#include <ATen/detail/XLAHooksInterface.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

namespace torch_xla::detail {

// XLA hooks implementation following PyTorch patterns
struct XLAHooks : public at::XLAHooksInterface {
  XLAHooks(const at::XLAHooksArgs& args) {}

  // Core accelerator interface methods
  void init() const override;
  bool hasXLA() const override;
  bool isAvailable() const override;
  bool isBuilt() const override { return true; }
  std::string showConfig() const override;

  // Device management
  c10::DeviceIndex deviceCount() const override;
  c10::DeviceIndex getCurrentDevice() const override;
  bool hasPrimaryContext(c10::DeviceIndex device_index) const override;

  // Memory management
  bool isPinnedPtr(const void* data) const override;
  c10::Allocator* getPinnedMemoryAllocator() const override;
  c10::Device getDeviceFromPtr(void* data) const override;

  // Generator methods
  const at::Generator& getDefaultGenerator(
      c10::DeviceIndex device_index = -1) const override;
  at::Generator getNewGenerator(
      c10::DeviceIndex device_index = -1) const override;
};

}  // namespace torch_xla::detail
