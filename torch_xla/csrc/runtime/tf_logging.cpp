#include "torch_xla/csrc/runtime/tf_logging.h"

#include <c10/util/Exception.h>

#include <stdexcept>

#include "tsl/platform/stacktrace.h"
#include <torch/csrc/utils/cpp_stacktraces.h>

#include "torch_xla/csrc/status.h"

namespace torch_xla {
namespace runtime {
namespace internal {

void ErrorGenerator::operator&(const std::basic_ostream<char>& oss) const {
  const ErrorSink& sink = dynamic_cast<const ErrorSink&>(oss);

  std::stringstream ess;
  ess << sink.str();

  if (torch::get_cpp_stacktraces_enabled()) {
    ess << " (at " << file_ << ":" << line_ << ")\n";
  }

  TF_VLOG(1) << ess.str();
  TORCH_CHECK(false, ess.str());
}

}  // namespace internal
}  // namespace runtime
}  // namespace torch_xla
