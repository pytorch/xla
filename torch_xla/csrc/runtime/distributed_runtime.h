#ifndef XLA_CLIENT_DISTRIBUTED_RUNTIME_H_
#define XLA_CLIENT_DISTRIBUTED_RUNTIME_H_

#include <memory>

#include "xla/pjrt/distributed/distributed.h"

namespace torch_xla {
namespace runtime {
  
class DistributedRuntime {
  public:
   DistributedRuntime(int global_rank);
   std::shared_ptr<xla::DistributedRuntimeClient> GetClient();

  private:
   std::unique_ptr<xla::DistributedRuntimeService> dist_runtime_service_;
   std::shared_ptr<xla::DistributedRuntimeClient> dist_runtime_client_;

   void shutdown();
};

}
}

#endif // XLA_CLIENT_DISTRIBUTED_RUNTIME_H_
