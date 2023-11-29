#include <torch/csrc/lazy/backend/backend_device.h>

#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/pjrt_computation_client.h"
#include "tsl/platform/stacktrace_handler.h"

namespace torch_xla {
namespace runtime {
namespace {

std::unique_ptr<ComputationClient> CreateClient() {
  if (sys_util::GetEnvBool("XLA_DUMP_FATAL_STACK", false)) {
    tsl::testing::InstallStacktraceHandler();
  }

  std::unique_ptr<ComputationClient> client = nullptr;

  if (sys_util::GetEnvString(env::kEnvPjRtDevice, "") != "") {
    client = std::make_unique<PjRtComputationClient>();
  } else {
    XLA_ERROR() << "$PJRT_DEVICE is not set." << std::endl;
  }

  XLA_CHECK(client);

  return client;
}

}  // namespace

ComputationClient* GetComputationClient(bool create = true) {
  static std::unique_ptr<ComputationClient> client = nullptr;
  if (!client && create) {
    static std::once_flag flag;
    std::call_once(flag, [](){ client = CreateClient(); });
  }
  return client.get();
}

ComputationClient* GetComputationClientIfInitialized() {
  return GetComputationClient(/*create=*/false);
}

}  // namespace runtime
}  // namespace torch_xla
