#ifndef XLA_TORCH_XLA_CSRC_XLA_BACKEND_IMPL_H_
#define XLA_TORCH_XLA_CSRC_XLA_BACKEND_IMPL_H_

#include <torch/csrc/lazy/backend/backend_interface.h>

#include <iostream>
#include <string>

#include "third_party/xla_client/computation_client.h"
#include "torch_xla/csrc/device.h"

namespace torch_xla {

torch::lazy::BackendImplInterface* GetXlaBackendImpl();

bool InitXlaBackend();

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_XLA_BACKEND_IMPL_H_