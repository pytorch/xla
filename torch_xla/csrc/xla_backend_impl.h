#ifndef XLA_TORCH_XLA_CSRC_XLA_BACKEND_IMPL_H_
#define XLA_TORCH_XLA_CSRC_XLA_BACKEND_IMPL_H_

#include <torch/csrc/lazy/backend/backend_interface.h>

#include <iostream>
#include <string>

#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/runtime/computation_client.h"

namespace torch_xla {

torch::lazy::BackendImplInterface* GetXlaBackendImpl();

bool InitXlaBackend();

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_XLA_BACKEND_IMPL_H_
