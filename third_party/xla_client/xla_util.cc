#include "tensorflow/compiler/xla/xla_client/xla_util.h"

#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace xrt_util {

using namespace tensorflow;

StatusOr<std::unique_ptr<HloModule>> CreateModuleFromProto(
    const HloModuleProto& proto, const DebugOptions& debug_options) {
  TF_ASSIGN_OR_RETURN(
      auto hlo_module_config,
      HloModule::CreateModuleConfigFromProto(proto, debug_options));
  return HloModule::CreateFromProto(proto, hlo_module_config);
}

StatusOr<string> GetComputationHloText(const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(auto hlo_module,
                      CreateModuleFromProto(computation.proto()));
  return hlo_module->ToString();
}

void CheckComputationStatus(const Status& status,
                            const XlaComputation& computation) {
  if (!status.ok()) {
    string hlo_text = GetComputationHloText(computation).ValueOrDie();
    XLA_LOG_LINES(ERROR, hlo_text);
    LOG(FATAL) << status;
  }
}

}  // namespace xrt_util
}  // namespace xla
