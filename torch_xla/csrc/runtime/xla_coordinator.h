#ifndef PTXLA_RUNTIME_COORDINATOR_H_
#define PTXLA_RUNTIME_COORDINATOR_H_

#include <memory>

#include "absl/base/nullability.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/tsl/distributed_runtime/preemption/preemption_sync_manager.h"

namespace torch_xla {
namespace runtime {

// XlaCoordinator serves as the point of entry for all operations which
// required the XLA distributed runtime, such as preemption coordination.
class XlaCoordinator {
 private:
  // Private struct for making the constructor private, but still callable
  // with std::make_unique<T>() function.
  struct PrivateUse {
    explicit PrivateUse() = default;
  };

 public:
  static inline const std::string kDefaultCoordinatorPort = "8547";

  XlaCoordinator(PrivateUse);

  ~XlaCoordinator();

  // Retrieve the DistributedRuntimeClient.
  std::shared_ptr<xla::DistributedRuntimeClient> GetClient();

  // Register a PreemptionSyncManager for the distributed runtime if none is
  // active. The PreemptionSyncManager will register a SIGTERM handler, and
  // when any host has received a preemption notice, all hosts are made aware
  // through the ReachedSyncPoint API. See the documentation of
  // tsl::PreemptionSyncManager for the full semantics:
  // https://github.com/google/tsl/blob/3bbe663/tsl/distributed_runtime/preemption/preemption_sync_manager.h#L34
  void ActivatePreemptionSyncManager();

  // If the PreemptionSyncManager is active, this will deactivate it and
  // destroy the current instance.
  void DeactivatePreemptionSyncManager();

  // A pass-through API to PreemptionSyncManager::ReachedSyncPoint.
  // The PreemptionSyncManager must be activated within the XlaCoordinator.
  // Returns true when the input step has been identified as a sync point, and
  // false otherwise.
  bool ReachedSyncPoint(int step);

  // Creates a new instance of XlaCoordinator, and initializes it.
  static absl::StatusOr<absl_nonnull std::unique_ptr<XlaCoordinator>> Create(
      int global_rank, int world_size, std::string master_addr,
      std::string port);

 private:
  // Convenience function called by `Create()` that initializes the current
  // XlaCoordinator.
  absl::Status Initialize(int global_rank, int world_size,
                          std::string master_addr, std::string port);

  std::unique_ptr<xla::DistributedRuntimeService> dist_runtime_service_;
  std::shared_ptr<xla::DistributedRuntimeClient> dist_runtime_client_;
  std::unique_ptr<tsl::PreemptionSyncManager> preemption_sync_manager_;
};

}  // namespace runtime
}  // namespace torch_xla

#endif  // PTXLA_RUNTIME_COORDINATOR_H_
