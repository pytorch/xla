#include "tensorflow/tsl/platform/stacktrace_handler.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/pjrt_computation_client.h"

#ifndef DISABLE_XRT
#include "torch_xla/csrc/runtime/xrt_computation_client.h"
#include "torch_xla/csrc/runtime/xrt_local_service.h"
#endif

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
#ifndef DISABLE_XRT
    client = new XrtComputationClient();
#else
    XLA_ERROR() << "$PJRT_DEVICE is not set." << std::endl;
#endif
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

void RunLocalService(uint64_t service_port) {
#ifndef DISABLE_XRT
  try {
    XrtLocalService* service = new XrtLocalService(
        "localservice|localhost:" + std::to_string(service_port),
        "localservice", 0);
    service->Start();
    service->Join();
  } catch (const std::runtime_error& error) {
    if (std::string(error.what()).find("Couldn't open device: /dev/accel0") !=
        std::string::npos) {
      TF_LOG(INFO) << "Local service has been created by other process, return";
    } else {
      throw;
    }
  }
#else
  XLA_ERROR() << "PyTorch/XLA was not built with XRT support." << std::endl;
#endif
}

}  // namespace runtime
}  // namespace torch_xla
