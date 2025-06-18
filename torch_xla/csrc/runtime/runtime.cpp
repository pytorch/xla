#include <torch/csrc/lazy/backend/backend_device.h>

#include "absl/log/absl_check.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/ifrt_computation_client.h"
#include "torch_xla/csrc/runtime/pjrt_computation_client.h"
#include "tsl/platform/stacktrace_handler.h"

namespace torch_xla::runtime {

std::atomic<bool> g_computation_client_initialized(false);

// Creates a new instance of a `ComputationClient` (e.g. `PjRtComputationClient`),
// and initializes the computation client
static absl::StatusOr<absl_nonnull std::unique_ptr<ComputationClient>>
InitializeComputationClient() {
  if (sys_util::GetEnvBool("XLA_DUMP_FATAL_STACK", false)) {
    tsl::testing::InstallStacktraceHandler();
  }

  std::unique_ptr<ComputationClient> client;

  // Disable IFRT right now as it currently crashes.
  // static bool use_ifrt = sys_util::GetEnvBool("XLA_USE_IFRT", false);
  static bool use_ifrt = false;
  if (sys_util::GetEnvString(env::kEnvPjRtDevice, "") != "") {
    if (use_ifrt) {
      client = std::make_unique<IfrtComputationClient>();
    } else {
      client = std::make_unique<PjRtComputationClient>();
    }
  } else {
    return absl::FailedPreconditionError("$PJRT_DEVICE is not set.");
  }

  g_computation_client_initialized = true;
  return client;
}

absl::StatusOr<ComputationClient*> GetComputationClient() {
  static absl::StatusOr<absl_nonnull std::unique_ptr<ComputationClient>> maybeClient = InitializeComputationClient();

  if (!maybeClient.ok()) {
    return maybeClient.status();
  }

  auto client = maybeClient.value().get();
  ABSL_CHECK(client);
  return client;
}

ComputationClient* GetComputationClientOrDie() {
  return GetComputationClient().value();
}

ComputationClient* GetComputationClientIfInitialized() {
  return g_computation_client_initialized ? GetComputationClientOrDie()
                                          : nullptr;
}

}  // namespace torch_xla::runtime
