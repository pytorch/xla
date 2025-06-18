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

static absl::StatusOr<std::unique_ptr<ComputationClient>>
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

  return client;
}

absl::StatusOr<ComputationClient*> GetComputationClient() {
  static std::unique_ptr<ComputationClient> client;

  if (client.get() == nullptr) {
    // Try to initialize the computation client if it's not
    // already initialized.
    auto status = InitializeComputationClient();

    if (status.ok()) {
      client = std::move(status.value());
      ABSL_CHECK(client.get());
      g_computation_client_initialized = true;
    } else {
      return status.status();
    }
  }

  return client.get();
}

}  // namespace status

ComputationClient* GetComputationClientOrDie() {
  auto client = GetComputationClient();

  // In order to be backward compatible, we call `XLA_CHECK()`, which throws an
  // exception.
  //
  // Calling either `ConsumeValue()`, `XLA_CHECK_OK()`, or `ABSL_CHECK()` would
  // crash the process.
  XLA_CHECK(client.ok()) << client.status().message();

  return client.value();
}

ComputationClient* GetComputationClientIfInitialized() {
  return g_computation_client_initialized ? GetComputationClientOrDie()
                                          : nullptr;
}

}  // namespace torch_xla::runtime
