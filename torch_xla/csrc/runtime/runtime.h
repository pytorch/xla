#ifndef XLA_CLIENT_RUNTIME_H_
#define XLA_CLIENT_RUNTIME_H_

#include <torch/csrc/lazy/backend/backend_device.h>

#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/runtime/computation_client.h"

namespace torch_xla {
namespace runtime {

// Returns the ComputationClient singleton.
ComputationClient* GetComputationClient();

ComputationClient* GetComputationClientIfInitialized();

}  // namespace runtime
}  // namespace torch_xla

#endif
