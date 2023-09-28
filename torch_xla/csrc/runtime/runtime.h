#ifndef XLA_CLIENT_RUNTIME_H_
#define XLA_CLIENT_RUNTIME_H_

#include <torch/csrc/lazy/backend/backend_device.h>

#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/runtime/computation_client.h"

namespace torch_xla {
namespace runtime {

// Returns the ComputationClient singleton.
ComputationClient* GetComputationClient();

ComputationClient* GetComputationClientIfInitialized();

const torch::lazy::BackendDevice* GetDefaultDevice();

torch::lazy::BackendDevice GetCurrentDevice();

torch::lazy::BackendDevice SetCurrentDevice(
    const torch::lazy::BackendDevice& device);

static inline torch::lazy::BackendDevice GetDeviceOrCurrent(
    const torch::lazy::BackendDevice* device) {
  return device != nullptr ? *device : GetCurrentDevice();
}

}  // namespace runtime
}  // namespace torch_xla

#endif
