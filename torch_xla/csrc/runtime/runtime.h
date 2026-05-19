#ifndef XLA_CLIENT_RUNTIME_H_
#define XLA_CLIENT_RUNTIME_H_

#include "absl/base/attributes.h"
#include "absl/status/statusor.h"

#include "torch_xla/csrc/runtime/computation_client.h"

namespace torch_xla::runtime {

// Returns the ComputationClient singleton.
const absl::StatusOr<ComputationClient * absl_nonnull>& GetComputationClient();

// Returns the ComputationClient singleton if it was successfully initialized.
// Returns a nullptr if the ComputationClient wasn't initialized yet.
// Throws an exception if the ComputationClient was initialized but the
// initialization failed.
ComputationClient* GetComputationClientIfInitialized();

// Runs the XRT local service, this will block the caller unitl the server
// being stopped.
void RunLocalService(uint64_t service_port);

}  // namespace torch_xla::runtime

#endif
