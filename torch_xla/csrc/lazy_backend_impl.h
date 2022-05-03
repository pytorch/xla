#pragma once

#include <iostream>
#include <string>

#include "torch/csrc/lazy/backend/backend_interface.h"

namespace torch_xla {

torch::lazy::BackendImplInterface* GetXlaBackendImpl();

void InitXlaBackend();

}  // namespace torch_xla