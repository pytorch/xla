#ifndef XLA_CLIENT_DISTRIBUTED_RUNTIME_H_
#define XLA_CLIENT_DISTRIBUTED_RUNTIME_H_

#include <memory>

#include "xla/pjrt/distributed/distributed.h"

namespace torch_xla {
namespace runtime {

xla::DistributedRuntimeClient* GetDistributedRuntimeClient(int global_rank);

class DistributedRuntime {
 public:
  DistributedRuntime(int global_rank);
  // static DistributedRuntime& getInstance(int global_rank) {
  //   static DistributedRuntime dist_runtime_instance(global_rank);
  //   return dist_runtime_instance;
  // }
  ~DistributedRuntime();
  // DistributedRuntime(DistributedRuntime const&) = delete;
  // void operator=(DistributedRuntime const&) = delete;

  std::shared_ptr<xla::DistributedRuntimeClient> GetClient();

 private:

  std::unique_ptr<xla::DistributedRuntimeService> dist_runtime_service_;
  std::shared_ptr<xla::DistributedRuntimeClient> dist_runtime_client_;

  void shutdown_service();
  void shutdown_client();
};

}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_DISTRIBUTED_RUNTIME_H_
