#include "torch_xla/csrc/computation.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

namespace torch_xla {

Computation::Computation(std::string name, xla::XlaComputation computation)
    : name_(std::move(name)), computation_(std::move(computation)) {
  program_shape_ = ConsumeValue(computation_.GetProgramShape());
  hash_ = xla::util::MHash(name_, computation_.proto().SerializeAsString());
}

}  // namespace torch_xla
