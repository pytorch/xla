#include "torch_xla/csrc/runtime/xla_coordinator.h"

#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/status.h"
#include "xla/pjrt/distributed/distributed.h"

namespace torch_xla {
namespace runtime {

absl::Status XlaCoordinator::Initialize(int global_rank, int world_size,
                                        std::string master_addr,
                                        std::string port) {
  std::string dist_service_addr = absl::StrJoin({master_addr, port}, ":");
  if (global_rank == 0) {
    xla::CoordinationServiceImpl::Options service_options;
    service_options.num_nodes = world_size;
    // Default value can be found in
    // https://github.com/openxla/xla/blob/4b88636002bc5834d7fe3f862997c66a490987bc/xla/pjrt/distributed/client.h#L63-L72.
    int heartbeat_interval_sec =
        sys_util::GetEnvInt(env::kEnvDistSvcHeartbeatIntervalInSec, 100);
    service_options.heartbeat_timeout = absl::Seconds(heartbeat_interval_sec);
    int shutdown_timeout =
        sys_util::GetEnvInt(env::kEnvDistSvcShutdownTimeoutInMin, 5);
    service_options.shutdown_timeout = absl::Minutes(shutdown_timeout);

    XLA_ASSIGN_OR_RETURN(
        dist_runtime_service_,
        xla::GetDistributedRuntimeService(dist_service_addr, service_options),
        "Failed to initialize distributed runtime service.");
  }

  xla::DistributedRuntimeClient::Options client_options;
  client_options.node_id = global_rank;
  dist_runtime_client_ =
      xla::GetDistributedRuntimeClient(dist_service_addr, client_options);
  XLA_RETURN_IF_ERROR(dist_runtime_client_->Connect(),
                      "Failed to initialize distributed runtime client");
  return absl::OkStatus();
}

absl::StatusOr<absl_nonnull std::unique_ptr<XlaCoordinator>>
XlaCoordinator::Create(int global_rank, int world_size, std::string master_addr,
                       std::string port) {
  auto coordinator = std::make_unique<XlaCoordinator>(PrivateUse());
  XLA_RETURN_IF_ERROR(
      coordinator->Initialize(global_rank, world_size, master_addr, port));
  return std::move(coordinator);
}

XlaCoordinator::XlaCoordinator(PrivateUse) {}

XlaCoordinator::~XlaCoordinator() {
  preemption_sync_manager_ = nullptr;
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

std::shared_ptr<xla::DistributedRuntimeClient> XlaCoordinator::GetClient() {
  XLA_CHECK(dist_runtime_client_ != nullptr)
      << "distributed runtime client is null.";
  return dist_runtime_client_;
}

void XlaCoordinator::ActivatePreemptionSyncManager() {
  if (preemption_sync_manager_ == nullptr) {
    preemption_sync_manager_ = std::move(tsl::CreatePreemptionSyncManager());
    auto client = dist_runtime_client_->GetCoordinationServiceAgent();
    XLA_CHECK(client.ok()) << "Failed to retrieve the CoodinationServiceAgent";
    auto status = preemption_sync_manager_->Initialize(client.value());
    XLA_CHECK(status.ok()) << "Failed to initialize the PreemptionSyncManager";
  }
}

void XlaCoordinator::DeactivatePreemptionSyncManager() {
  preemption_sync_manager_ = nullptr;
}

bool XlaCoordinator::ReachedSyncPoint(int step) {
  XLA_CHECK(preemption_sync_manager_ != nullptr)
      << "A PreemptionSyncManager has not been registered with the "
         "XlaCoordinator.";
  return preemption_sync_manager_->ReachedSyncPoint(step);
}

}  // namespace runtime
}  // namespace torch_xla
