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
}  // namespace torch_xla
