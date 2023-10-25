#include "torch_xla/csrc/runtime/distributed_runtime.h"

#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/sys_util.h"

namespace torch_xla {
namespace runtime {

DistributedRuntime::DistributedRuntime(int global_rank, int world_size,
                                       std::string master_addr,
                                       std::string port) {
  std::string dist_service_addr = absl::StrJoin({master_addr, port}, ":");
  if (global_rank == 0) {
    xla::CoordinationServiceImpl::Options service_options;
    service_options.num_nodes = world_size;
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

void DistributedRuntime::ActivatePreemptionSyncManager() {
  if (preemption_sync_manager_ == nullptr) {
    preemption_sync_manager_ = std::move(tsl::CreatePreemptionSyncManager());
    auto client = dist_runtime_client_->GetCoordinationServiceAgent();
    XLA_CHECK(client.ok()) << "Failed to retrieve the CoodinationServiceAgent";
    auto status = preemption_sync_manager_->Initialize(client.value());
    XLA_CHECK(status.ok()) << "Failed to initialize the PreemptionSyncManager";
  }
}

bool DistributedRuntime::ReachedSyncPoint(int step) {
  XLA_CHECK(preemption_sync_manager_ != nullptr)
      << "A PreemptionSyncManager has not been registered with the "
         "DistributedRuntime.";
  return preemption_sync_manager_->ReachedSyncPoint(step);
}

}  // namespace runtime
}  // namespace torch_xla
