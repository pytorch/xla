#include "tensorflow/compiler/xla/xla_client/tf_logging.h"

#include <stdexcept>

namespace xla {
namespace internal {

void ErrorGenerator::operator&(const std::basic_ostream<char>& oss) const {
  const ErrorSink& sink = dynamic_cast<const ErrorSink&>(oss);
  auto sink_str = sink.str();
  TF_LOG(ERROR) << sink_str;
  std::stringstream ess;
  ess << file_ << ":" << line_ << " : " << sink_str;
  // We cannot use AT_ERROR() here, due to layering issues.
  throw std::runtime_error(ess.str());
}

}  // namespace internal
}  // namespace xla
