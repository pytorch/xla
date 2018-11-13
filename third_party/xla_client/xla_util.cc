#include "tensorflow/compiler/xla/xla_client/xla_util.h"

#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stacktrace.h"

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

void CheckComputationStatus(
    const Status& status,
    tensorflow::gtl::ArraySlice<const XlaComputation* const> computations) {
  if (!status.ok()) {
    for (size_t i = 0; i < computations.size(); ++i) {
      string hlo_text = GetComputationHloText(*computations[i]).ValueOrDie();
      LOG(ERROR) << ">>> Dumping Computation " << i;
      XLA_LOG_LINES(ERROR, hlo_text);
    }
    LOG(ERROR) << "StackTrace:\n" << tensorflow::CurrentStackTrace();
    LOG(FATAL) << "Status Error: " << status;
  }
}

}  // namespace xrt_util
}  // namespace xla
