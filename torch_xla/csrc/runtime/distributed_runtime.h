#ifndef XLA_CLIENT_DISTRIBUTED_RUNTIME_H_
#define XLA_CLIENT_DISTRIBUTED_RUNTIME_H_

#include <memory>

#include "xla/pjrt/distributed/distributed.h"

namespace torch_xla {
namespace runtime {

class DistributedRuntime {
 public:
  static const std::string default_coordinator_port;
  static DistributedRuntime& getInstance(int global_rank,
                                         std::string master_addr,
                                         std::string port) {
    static DistributedRuntime dist_runtime_instance(global_rank, master_addr,
                                                    port);
    return dist_runtime_instance;
  }
  ~DistributedRuntime();
  DistributedRuntime(DistributedRuntime const&) = delete;
  void operator=(DistributedRuntime const&) = delete;

  std::shared_ptr<xla::DistributedRuntimeClient> GetClient();

 private:
  DistributedRuntime(int global_rank, std::string master_addr,
                     std::string port);

  std::unique_ptr<xla::DistributedRuntimeService> dist_runtime_service_;
  std::shared_ptr<xla::DistributedRuntimeClient> dist_runtime_client_;
};

}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_DISTRIBUTED_RUNTIME_H_
