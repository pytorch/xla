#include "torch_xla/csrc/runtime/distributed_runtime.h"

#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/sys_util.h"

namespace torch_xla {
namespace runtime {

const std::string DistributedRuntime::default_coordinator_port = "8547";

DistributedRuntime::DistributedRuntime(int global_rank, std::string master_addr,
                                       std::string port) {
  std::string dist_service_addr = absl::StrJoin({master_addr, port}, ":");
  if (global_rank == 0) {
    int local_world_size = sys_util::GetEnvInt("LOCAL_WORLD_SIZE", 1);
    int global_world_size = sys_util::GetEnvInt("WORLD_SIZE", local_world_size);
    xla::CoordinationServiceImpl::Options service_options;
    service_options.num_nodes = global_world_size;
    xla::StatusOr<std::unique_ptr<xla::DistributedRuntimeService>>
        dist_runtime_service = xla::GetDistributedRuntimeService(
            dist_service_addr, service_options);
    XLA_CHECK(dist_runtime_service.ok())
        << "Failed to initialize distributed runtime service.";
    dist_runtime_service_ = std::move(dist_runtime_service.value());
  }

  xla::DistributedRuntimeClient::Options client_options;
  client_options.node_id = global_rank;
  dist_runtime_client_ =
      xla::GetDistributedRuntimeClient(dist_service_addr, client_options);
  XLA_CHECK(dist_runtime_client_->Connect().ok())
      << "Failed to initialize distributed runtime client";
}

DistributedRuntime::~DistributedRuntime() {
  if (dist_runtime_client_ != nullptr) {
    XLA_CHECK(dist_runtime_client_->Shutdown().ok())
        << "Failed to shut down the distributed runtime client.";
    dist_runtime_client_ = nullptr;
  }
  if (dist_runtime_service_ != nullptr) {
    dist_runtime_service_->Shutdown();
    dist_runtime_service_ = nullptr;
  }
}

std::shared_ptr<xla::DistributedRuntimeClient> DistributedRuntime::GetClient() {
  XLA_CHECK(dist_runtime_client_ != nullptr)
      << "distributed runtime client is null.";
  return dist_runtime_client_;
}

}  // namespace runtime
}  // namespace torch_xla
