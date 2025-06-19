#include "torch_xla/csrc/status.h"

#include "torch_xla/csrc/runtime/sys_util.h"

namespace torch_xla {

void ConsumeAndMaybeThrow(absl::Status status) {
  if (!status.ok()) {
    throw std::runtime_error(std::string(status.message()));
  }
}

}

