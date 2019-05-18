#include "tensorflow/compiler/xla/xla_client/xla_util.h"

#include <stdexcept>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/stacktrace.h"

namespace xla {
namespace util {
namespace {

size_t SingleShapeHash(const Shape& shape, size_t seed) {
  for (auto dim : shape.layout().minor_to_major()) {
    seed = HashCombine(seed, dim);
  }
  for (auto dim : shape.dimensions()) {
    seed = HashCombine(seed, dim);
  }
  return HashCombine(seed, static_cast<int>(shape.element_type()));
}

}  // namespace

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

void ReportComputationError(
    const Status& status,
    tensorflow::gtl::ArraySlice<const XlaComputation* const> computations) {
  for (size_t i = 0; i < computations.size(); ++i) {
    string hlo_text = GetComputationHloText(*computations[i]).ValueOrDie();
    TF_LOG(ERROR) << ">>> Dumping Computation " << i;
    XLA_LOG_LINES(tensorflow::ERROR, hlo_text);
  }
  TF_LOG(ERROR) << "StackTrace:\n" << tensorflow::CurrentStackTrace();
  TF_LOG(ERROR) << "Status: " << status;
  throw std::runtime_error(status.ToString());
}

void CheckComputationStatus(
    const Status& status,
    tensorflow::gtl::ArraySlice<const XlaComputation* const> computations) {
  if (!status.ok()) {
    ReportComputationError(status, computations);
  }
}

size_t ShapeHash(const Shape& shape) {
  size_t hash = 0xa5d2d6916;
  ShapeUtil::ForEachSubshape(shape,
                             [&](const Shape& subshape, const ShapeIndex&) {
                               hash = SingleShapeHash(subshape, hash);
                             });
  return hash;
}

}  // namespace util
}  // namespace xla
