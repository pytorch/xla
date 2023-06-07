#include "tsl/platform/stacktrace_handler.h"
#include "third_party/xla_client/computation_client.h"
#include "third_party/xla_client/env_vars.h"
#include "third_party/xla_client/pjrt_computation_client.h"
#include "third_party/xla_client/xrt_computation_client.h"
#include "third_party/xla_client/xrt_local_service.h"

namespace xla {
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
    client = new XrtComputationClient();
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
}

}  // namespace xla
