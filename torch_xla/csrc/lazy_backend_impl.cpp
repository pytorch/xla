#include "torch_xla/csrc/lazy_backend_impl.h"

namespace torch_xla {

class XlaBackendImpl : public torch::lazy::BackendImplInterface {
  public:
  
}

torch::lazy::BackendImplInterface* GetXlaBackendImpl() {
  static XlaBackendImpl* xla_backend_impl = new XlaBackendImpl();
  return xla_backend_impl;
}

void InitXlaBackend() {

}

}  // namespace torch_xla