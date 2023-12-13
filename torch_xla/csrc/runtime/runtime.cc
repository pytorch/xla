#include <torch/csrc/lazy/backend/backend_device.h>

#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/pjrt_computation_client.h"
#include "tsl/platform/stacktrace_handler.h"

namespace torch_xla {
namespace runtime {
namespace {

std::atomic<bool> g_computation_client_initialized(false);

ComputationClient* CreateClient() {
  bool was_initialized = g_computation_client_initialized.exchange(true);
  XLA_CHECK(!was_initialized) << "ComputationClient already initialized";
  if (sys_util::GetEnvBool("XLA_DUMP_FATAL_STACK", false)) {
    tsl::testing::InstallStacktraceHandler();
  }

  ComputationClient* client;

  if (sys_util::GetEnvString(env::kEnvPjRtDevice, "") != "") {
    client = new PjRtComputationClient();
  } else {
    g_computation_client_initialized = false;
    XLA_ERROR() << "$PJRT_DEVICE is not set." << std::endl;
  }

  XLA_CHECK(client != nullptr);

  return client;
}

}  // namespace

ComputationClient* GetComputationClient() {
  static auto client = std::unique_ptr<ComputationClient>(CreateClient());
  return client.get();
}

ComputationClient* GetComputationClientIfInitialized() {
  return g_computation_client_initialized ? GetComputationClient() : nullptr;
}

}  // namespace runtime
}  // namespace torch_xla
