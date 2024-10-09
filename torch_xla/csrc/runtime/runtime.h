#ifndef XLA_CLIENT_RUNTIME_H_
#define XLA_CLIENT_RUNTIME_H_

#include "torch_xla/csrc/runtime/computation_client.h"

namespace torch_xla {
namespace runtime {

// Returns the ComputationClient singleton.
ComputationClient* GetComputationClient();

ComputationClient* GetComputationClientIfInitialized();

void SetVirtualTopology(const std::string& topology);

// Run the XRT local service, this will block the caller unitl the server
// being stopped.
void RunLocalService(uint64_t service_port);

}  // namespace runtime
}  // namespace torch_xla

#endif
