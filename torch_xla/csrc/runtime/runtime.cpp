#include "torch_xla/csrc/runtime/runtime.h"

#include <torch/csrc/lazy/backend/backend_device.h>

#include "absl/log/absl_check.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/ifrt_computation_client.h"
#include "torch_xla/csrc/runtime/pjrt_computation_client.h"
#include "torch_xla/csrc/status.h"
#include "tsl/platform/stacktrace_handler.h"

namespace torch_xla::runtime {

static std::atomic<bool> g_computation_client_initialized(false);

// Creates a new instance of a `ComputationClient` (e.g.
// `PjRtComputationClient`), and initializes the computation client.
// Can only be called when g_computation_client_initialized is false.
static absl::StatusOr<ComputationClient * absl_nonnull>
InitializeComputationClient() {
  if (sys_util::GetEnvBool("XLA_DUMP_FATAL_STACK", false)) {
    tsl::testing::InstallStacktraceHandler();
  }

  // TODO: enable IFRT once it's not crashing anymore.
  // Ref: https://github.com/pytorch/xla/pull/8267
  //
  // static bool use_ifrt = sys_util::GetEnvBool("XLA_USE_IFRT", false);
  const bool use_ifrt = false;
  if (sys_util::GetEnvString(env::kEnvPjRtDevice, "") == "") {
    return XLA_ERROR_WITH_LOCATION(
        absl::FailedPreconditionError("$PJRT_DEVICE is not set."));
  }

  ABSL_CHECK(!g_computation_client_initialized)
      << "ComputationClient can only be initialized once.";

  std::unique_ptr<ComputationClient> client;
  if (use_ifrt) {
    XLA_ASSIGN_OR_RETURN(client, IfrtComputationClient::Create());
  } else {
    XLA_ASSIGN_OR_RETURN(client, PjRtComputationClient::Create());
  }

  // Set only if we actually successfully initialized a client.
  g_computation_client_initialized = true;

  return client.release();
}

const absl::StatusOr<ComputationClient * absl_nonnull>& GetComputationClient() {
  // Reference to singleton Status-wrapped ComputationClient instance.
  //
  // Since we only allow a single initialization, as soon as this function is
  // called, we store the initialization result in this trivially destructible
  // reference.
  static const auto& maybe_client =
      *new absl::StatusOr<ComputationClient*>(InitializeComputationClient());
  return maybe_client;
}

ComputationClient* absl_nonnull GetComputationClientOrDie() {
  XLA_ASSIGN_OR_THROW(ComputationClient * client, GetComputationClient());
  return client;
}

ComputationClient* GetComputationClientIfInitialized() {
  return g_computation_client_initialized ? GetComputationClientOrDie()
                                          : nullptr;
}

}  // namespace torch_xla::runtime
