#include "torch_xla/csrc/runtime/tf_logging.h"

#include <stdexcept>

#include "torch_xla/csrc/status.h"

namespace torch_xla {
namespace runtime {
namespace internal {

void ErrorGenerator::operator&(const std::basic_ostream<char>& oss) const {
  const ErrorSink& sink = dynamic_cast<const ErrorSink&>(oss);

  std::stringstream ess;
  ess << sink.str();

  if (ShouldShowCppErrorContext()) {
    ess << " (at " << file_ << ":" << line_ << ")";
  }

  TF_VLOG(1) << ess.str();
  // We cannot use AT_ERROR() here, due to layering issues.
  throw std::runtime_error(ess.str());
}

}  // namespace internal
}  // namespace runtime
}  // namespace torch_xla
