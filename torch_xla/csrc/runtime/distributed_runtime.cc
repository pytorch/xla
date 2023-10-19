#include <stdlib.h>

#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/distributed_runtime.h"
#include "torch_xla/csrc/runtime/sys_util.h"

namespace torch_xla {
namespace runtime {

DistributedRuntime::DistributedRuntime(int global_rank) {
  std::string master_addr =
          runtime::sys_util::GetEnvString("MASTER_ADDR", "localhost");
  std::string port =
      runtime::sys_util::GetEnvString("XLA_COORDINATOR_PORT", "8547");
  std::string dist_service_addr = absl::StrJoin({master_addr, port}, ":");
  if (global_rank == 0) {
    XLA_CHECK(!sys_util::GetEnvString("WORLD_SIZE", "").empty() || !sys_util::GetEnvString("LOCAL_WORLD_SIZE", "").empty()) << "WORLD_SIZE and LOCAL_WORLD_SIZE should be empty at the same time.";
    int global_world_size = sys_util::GetEnvInt("WORLD_SIZE", sys_util::GetEnvInt("LOCAL_WORLD_SIZE", -1));
    XLA_CHECK(global_world_size > 0) << "WORLD_SIZE and LOCAL_WORLD_SIZE should not be empty at the same time.";
    xla::CoordinationServiceImpl::Options service_options;
    service_options.num_nodes = global_world_size;
    XLA_CHECK(xla::GetDistributedRuntimeService(dist_service_addr, service_options).ok()) << "Failed to initialize distributed runtime service.";
    dist_runtime_service_ = xla::GetDistributedRuntimeService(dist_service_addr, service_options).value();
  }

  xla::DistributedRuntimeClient::Options client_options;
  client_options.node_id = global_rank;
  dist_runtime_client_ = xla::GetDistributedRuntimeClient(dist_service_addr, client_options);
  XLA_CHECK(dist_runtime_client_->Connect().ok())
      << "Failed to initialize distributed runtime client";
  // atexit(shutdown)
  // XLA_CHECK(atexit(shutdown) == 0);
}

std::shared_ptr<xla::DistributedRuntimeClient> DistributedRuntime::GetClient() {
  XLA_CHECK(dist_runtime_client_ != nullptr) << "dist_runtime_client_ has not been initialized yet.";
  return dist_runtime_client_;
}

void DistributedRuntime::shutdown() {
  if (dist_runtime_client_ != nullptr) {
    XLA_CHECK(dist_runtime_client_->Shutdown().ok()) << "Failed to shut down the distributed runtime client.";
  }
  if (dist_runtime_service_ != nullptr) {
    dist_runtime_service_->Shutdown();
  }
}

}
}
