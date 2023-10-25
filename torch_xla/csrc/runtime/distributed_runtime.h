#ifndef XLA_CLIENT_DISTRIBUTED_RUNTIME_H_
#define XLA_CLIENT_DISTRIBUTED_RUNTIME_H_

#include <memory>

#include "torch_xla/csrc/runtime/debug_macros.h"
#include "tsl/distributed_runtime/preemption/preemption_sync_manager.h"
#include "xla/pjrt/distributed/distributed.h"

namespace torch_xla {
namespace runtime {

// DistributedRuntime serves as the point of entry for all operations which
// required the XLA distributed runtime, such as preemption coordination.
class DistributedRuntime {
 public:
  static inline const std::string kDefaultCoordinatorPort = "8547";

  // Returns true if the distributed runtime has already been initialized.
  static bool IsInitialized() { return dist_runtime_ != nullptr; }

  // Initialize the shared DistributedRuntime object. This creates a
  // DistributedRuntimeClient on each worker, and on global_rank 0 initializes
  // the corresponding DistributedRuntimeService.
  static void Initialize(int global_rank, int world_size,
                         std::string master_addr, std::string port) {
    XLA_CHECK(!IsInitialized()) << "DistributedRuntime already initialized";
    dist_runtime_ = std::unique_ptr<DistributedRuntime>(
        new DistributedRuntime(global_rank, world_size, master_addr, port));
  }

  // Shutdown the distributed runtime. All associated resources will be
  // released, and subsequent calls to IsInitialized will return false.
  // The distributed runtime may later be re-initialized.
  static void Shutdown() {
    XLA_CHECK(IsInitialized())
        << "Must initialize distributed runtime before shutdown";
    dist_runtime_ = nullptr;
  }

  // Retrieve the shared DistributedRuntime object.
  static DistributedRuntime& Get() {
    XLA_CHECK(IsInitialized())
        << "Must initialize distributed runtime before retrieval";
    return *dist_runtime_;
  }

  ~DistributedRuntime();
  DistributedRuntime(DistributedRuntime const&) = delete;
  void operator=(DistributedRuntime const&) = delete;

  // Retrieve the DistributedRuntimeClient.
  std::shared_ptr<xla::DistributedRuntimeClient> GetClient();

  // Register a PreemptionSyncManager for the distributed runtime if none is
  // active. The PreemptionSyncManager will register a SIGTERM handler, and
  // when any host has received a preemption notice, all hosts are made aware
  // through the ReachedSyncPoint API. See the documentation of
  // tsl::PreemptionSyncManager for the full semantics:
  // https://github.com/google/tsl/blob/3bbe663/tsl/distributed_runtime/preemption/preemption_sync_manager.h#L34
  void ActivatePreemptionSyncManager();

  // A pass-throguh API to the PreemptionSyncManager::ReachedSyncPoint.
  // The PreemptionSyncManager must be activated within the DistributedRuntime.
  // Returns true when the input step has been identified as a sync point, and
  // false otherwise.
  bool ReachedSyncPoint(int step);

 private:
  static inline std::unique_ptr<DistributedRuntime> dist_runtime_;

  DistributedRuntime(int global_rank, int world_size, std::string master_addr,
                     std::string port);

  std::unique_ptr<xla::DistributedRuntimeService> dist_runtime_service_;
  std::shared_ptr<xla::DistributedRuntimeClient> dist_runtime_client_;
  std::shared_ptr<tsl::PreemptionSyncManager> preemption_sync_manager_;
};

}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_DISTRIBUTED_RUNTIME_H_
