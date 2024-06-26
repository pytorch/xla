#include <torch/csrc/lazy/backend/backend_device.h>

#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/ifrt_computation_client.h"
#include "torch_xla/csrc/runtime/pjrt_compilation_client.h"
#include "torch_xla/csrc/runtime/pjrt_computation_client.h"
#include "tsl/platform/stacktrace_handler.h"

namespace torch_xla {
namespace runtime {

std::atomic<bool> g_computation_client_initialized(false);

std::string aot_topology = "";

ComputationClient* GetComputationClient() {
  static std::unique_ptr<ComputationClient> client = []() {
    if (sys_util::GetEnvBool("XLA_DUMP_FATAL_STACK", false)) {
      tsl::testing::InstallStacktraceHandler();
    }

    std::unique_ptr<ComputationClient> client;

    static bool use_ifrt = sys_util::GetEnvBool("XLA_USE_IFRT", false);
    if (sys_util::GetEnvString(env::kEnvPjRtDevice, "") != "") {
      if (aot_topology != "") {
        // aot returns nullptr client
        client = std::make_unique<PjRtCompilationClient>(aot_topology);
      } else if (use_ifrt) {
        client = std::make_unique<IfrtComputationClient>();
      } else {
        client = std::make_unique<PjRtComputationClient>();
      }
    } else {
      XLA_ERROR() << "$PJRT_DEVICE is not set." << std::endl;
    }

    XLA_CHECK(client);

    g_computation_client_initialized = true;
    return client;
  }();

  return client.get();
}

void SetVirtualTopology(const std::string& topology) {
  XLA_CHECK(!g_computation_client_initialized)
      << "AOT topology must be specified before computation client "
         "initialization.";
  aot_topology = topology;
}

ComputationClient* GetComputationClientIfInitialized() {
  return g_computation_client_initialized ? GetComputationClient() : nullptr;
}

}  // namespace runtime
}  // namespace torch_xla
