#ifndef XLA_CLIENT_RUNTIME_H_
#define XLA_CLIENT_RUNTIME_H_

#include "third_party/xla_client/computation_client.h"

namespace xla {

// Returns the ComputationClient singleton.
ComputationClient* GetClient();

ComputationClient* GetClientIfInitialized();

// Run the XRT local service, this will block the caller unitl the server
// being stopped.
void RunLocalService(uint64_t service_port);

}  // namespace xla

#endif
