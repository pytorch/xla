#ifndef XLA_CLIENT_RUNTIME_H_
#define XLA_CLIENT_RUNTIME_H_

#include "torch_xla/csrc/runtime/computation_client.h"

namespace torch_xla::runtime {
namespace status {

// Returns the ComputationClient singleton.
absl::StatusOr<ComputationClient*> GetComputationClient();

}

ABSL_DEPRECATED(
    "Use status::GetComputationClient(), instead. "
    "This function throws an exception on error, instead of "
    "actually handling the StatusOr return value, which is "
    "safer.")
ComputationClient* GetComputationClient();

ComputationClient* GetComputationClientIfInitialized();

// Run the XRT local service, this will block the caller unitl the server
// being stopped.
void RunLocalService(uint64_t service_port);

}  // namespace torch_xla::runtime

#endif
