#ifndef XLA_CLIENT_RUNTIME_H_
#define XLA_CLIENT_RUNTIME_H_

#include "torch_xla/csrc/runtime/computation_client.h"

namespace xla {

// Returns the ComputationClient singleton.
ComputationClient* GetComputationClient();

ComputationClient* GetComputationClientIfInitialized();

// Run the XRT local service, this will block the caller unitl the server
// being stopped.
void RunLocalService(uint64_t service_port);

}  // namespace xla

#endif
