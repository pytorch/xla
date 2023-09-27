#include <torch/csrc/lazy/backend/backend_device.h>

#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/pjrt_computation_client.h"
#include "tsl/platform/stacktrace_handler.h"

namespace torch_xla {
namespace runtime {
namespace {

std::atomic<ComputationClient*> g_computation_client(nullptr);
std::once_flag g_computation_client_once;


ComputationClient* CreateClient() {
  if (sys_util::GetEnvBool("XLA_DUMP_FATAL_STACK", false)) {
    tsl::testing::InstallStacktraceHandler();
  }

  ComputationClient* client;

  if (sys_util::GetEnvString(env::kEnvPjRtDevice, "") != "") {
    client = new PjRtComputationClient();
  } else {
    XLA_ERROR() << "$PJRT_DEVICE is not set." << std::endl;
  }

  XLA_CHECK(client != nullptr);

  return client;
}

}  // namespace

ComputationClient* GetComputationClient() {
  std::call_once(g_computation_client_once,
                 [&]() { g_computation_client = std::move(CreateClient()); });
  return g_computation_client.load();
}

ComputationClient* GetComputationClientIfInitialized() {
  return g_computation_client.load();
}

}  // namespace runtime

namespace {

thread_local absl::optional<torch::lazy::BackendDevice> g_current_device;

}

const torch::lazy::BackendDevice* GetDefaultDevice() {
  std::string default_device_spec =
      UseVirtualDevice()
            ? "SPMD:0"
            : runtime::GetComputationClient()->GetDefaultDevice();
  XLA_CHECK(!default_device_spec.empty());
  static const torch::lazy::BackendDevice default_device =
      ParseDeviceString(default_device_spec);
  return &default_device;
}

torch::lazy::BackendDevice GetCurrentDevice() {
  if (!g_current_device) {
    g_current_device = *GetDefaultDevice();
  }
  return *g_current_device;
}


torch::lazy::BackendDevice SetCurrentDevice(
    const torch::lazy::BackendDevice& device) {
  torch::lazy::BackendDevice current = GetCurrentDevice();
  g_current_device = device;
  TF_VLOG(2) << "New current device: " << device;
  return current;
}

}  // namespace torch_xla
