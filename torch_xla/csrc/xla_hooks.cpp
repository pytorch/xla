#include "xla_hooks.h"

#include <sstream>
#include <iostream>

// PyTorch integration headers
#include <ATen/core/Generator.h>
#include <ATen/detail/XLAHooksInterface.h>
#include <c10/util/Exception.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/Logging.h>

// XLA headers
#include "xla_generator.h"
#include "xla_backend_impl.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/runtime.h"


namespace torch_xla::detail {

void XLAHooks::init() const {
  C10_LOG_API_USAGE_ONCE("aten.init.xla");
  
  // Initialize XLA backend - this registers XLA functions and sets up 
  // the backend infrastructure
  torch_xla::InitXlaBackend();
}

bool XLAHooks::hasXLA() const {
  return isAvailable();
}

bool XLAHooks::isAvailable() const {
  try {
    return deviceCount() > 0;
  } catch (...) {
    // If device enumeration fails, XLA is not available
    return false;
  }
}

std::string XLAHooks::showConfig() const {
  std::ostringstream oss;
  oss << "XLA Backend Configuration:\n";
  oss << "  - XLA devices available: " << deviceCount() << "\n";
  return oss.str();
}

c10::DeviceIndex XLAHooks::deviceCount() const {
  auto maybe_client = torch_xla::runtime::GetComputationClient();
  if (!maybe_client.ok()) {
    // If runtime client initialization failed, return 0 devices
    return 0;
  }
  
  auto* client = maybe_client.value();
  return static_cast<c10::DeviceIndex>(client->GetNumDevices());
}

c10::DeviceIndex XLAHooks::getCurrentDevice() const {
  return bridge::GetCurrentAtenDevice().index();
}

bool XLAHooks::hasPrimaryContext(c10::DeviceIndex device_index) const {
  TORCH_CHECK(false, "hasPrimaryContext is not implemented.");
}

bool XLAHooks::isPinnedPtr(const void* data) const {
  TORCH_CHECK(false, "isPinnedPtr is not implemented.");
}

c10::Allocator* XLAHooks::getPinnedMemoryAllocator() const {
  TORCH_CHECK(false, "getPinnedMemoryAllocator is not implemented.");
}

c10::Device XLAHooks::getDeviceFromPtr(void* data) const {
  TORCH_CHECK(false, "getDeviceFromPtr is not implemented.");
}

const at::Generator& XLAHooks::getDefaultGenerator(c10::DeviceIndex device_index) const {
  return at::detail::getDefaultXLAGenerator(device_index);
}

at::Generator XLAHooks::getNewGenerator(c10::DeviceIndex device_index) const {
  // Create and return a new XLA generator using the make_generator template function
  return at::make_generator<at::XLAGeneratorImpl>(device_index);
}

} // namespace torch_xla::detail

// Register XLA hooks with PyTorch on module load
namespace at {
REGISTER_XLA_HOOKS(torch_xla::detail::XLAHooks)
} // namespace at
