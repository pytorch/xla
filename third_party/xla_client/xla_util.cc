#include "tensorflow/compiler/xla/xla_client/xla_util.h"

#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/stacktrace.h"

namespace xla {
namespace util {
namespace {

hash_t SingleShapeHash(const Shape& shape, hash_t seed) {
  if (shape.has_layout()) {
    for (auto dim : shape.layout().minor_to_major()) {
      seed = HashCombine(seed, dim);
    }
  }
  for (auto dim : shape.dimensions()) {
    seed = HashCombine(seed, dim);
  }
  return HashCombine(seed, static_cast<int>(shape.element_type()));
}

void MaybeSaveHloGraph(const std::string& hlo_text, size_t index) {
  static const std::string save_file =
      sys_util::GetEnvString("XLA_SAVE_HLO_FILE", "");
  if (!save_file.empty()) {
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    std::ofstream graph_file(save_file, std::ios_base::app);
    graph_file << "[HLO Graph " << index << " From Thread "
               << std::this_thread::get_id() << "]\n"
               << hlo_text << "\n";
  }
}

std::string MaybeDumpHloGraph(
    const absl::Span<const Shape* const>& output_shapes,
    const std::string& hlo_text, size_t index) {
  static const bool dump_hlo =
      sys_util::GetEnvBool("XLA_DUMP_HLO_GRAPH", false);
  if (!dump_hlo) {
    return {};
  }
  std::stringstream ss;
  ss << ">>> Dumping Computation " << index << "\n";
  ss << hlo_text << "\n";
  if (index < output_shapes.size() && output_shapes[index] != nullptr) {
    ss << "OutputShape: " << *output_shapes[index] << "\n\n";
  }
  return ss.str();
}

}  // namespace

StatusOr<std::unique_ptr<HloModule>> CreateModuleFromProto(
    const HloModuleProto& proto, const DebugOptions& debug_options) {
  TF_ASSIGN_OR_RETURN(
      auto hlo_module_config,
      HloModule::CreateModuleConfigFromProto(proto, debug_options));
  return HloModule::CreateFromProto(proto, hlo_module_config);
}

StatusOr<std::string> GetComputationHloText(const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(auto hlo_module,
                      CreateModuleFromProto(computation.proto()));
  return hlo_module->ToString();
}

void ReportComputationError(
    const Status& status, absl::Span<const XlaComputation* const> computations,
    absl::Span<const Shape* const> output_shapes) {
  std::stringstream ss;
  for (size_t i = 0; i < computations.size(); ++i) {
    std::string hlo_text = GetComputationHloText(*computations[i]).value();
    MaybeSaveHloGraph(hlo_text, i);
    ss << MaybeDumpHloGraph(output_shapes, hlo_text, i);
  }
  ss << "StackTrace:\n" << tensorflow::CurrentStackTrace() << "\n";
  ss << "Status: " << status << "\n";
  XLA_LOG_LINES(tensorflow::ERROR, ss.str());
  throw std::runtime_error(status.ToString());
}

void CheckComputationStatus(
    const Status& status, absl::Span<const XlaComputation* const> computations,
    absl::Span<const Shape* const> output_shapes) {
  if (!status.ok()) {
    ReportComputationError(status, computations, output_shapes);
  }
}

hash_t ShapeHash(const Shape& shape) {
  hash_t hash = 0xa5d2d6916;
  ShapeUtil::ForEachSubshape(shape,
                             [&](const Shape& subshape, const ShapeIndex&) {
                               hash = SingleShapeHash(subshape, hash);
                             });
  return hash;
}

}  // namespace util
}  // namespace xla
