#include "torch_xla/csrc/runtime/xla_util.h"

#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/stacktrace.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch_xla/csrc/runtime/util.h"

namespace torch_xla {
namespace runtime {
namespace util {
namespace {

hash_t SingleShapeHash(const xla::Shape& shape, hash_t seed) {
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
    const absl::Span<const xla::Shape* const>& output_shapes,
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

xla::StatusOr<std::unique_ptr<xla::HloModule>> CreateModuleFromProto(
    const xla::HloModuleProto& proto, const xla::DebugOptions& debug_options) {
  TF_ASSIGN_OR_RETURN(
      auto hlo_module_config,
      xla::HloModule::CreateModuleConfigFromProto(proto, debug_options));
  return xla::HloModule::CreateFromProto(proto, hlo_module_config);
}

xla::StatusOr<std::string> GetComputationHloText(
    const xla::XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(auto hlo_module,
                      CreateModuleFromProto(computation.proto()));
  return hlo_module->ToString();
}

void ReportComputationError(
    const xla::Status& status,
    absl::Span<const xla::XlaComputation* const> computations,
    absl::Span<const xla::Shape* const> output_shapes) {
  std::stringstream ss;
  for (size_t i = 0; i < computations.size(); ++i) {
    std::string hlo_text = GetComputationHloText(*computations[i]).value();
    MaybeSaveHloGraph(hlo_text, i);
    ss << MaybeDumpHloGraph(output_shapes, hlo_text, i);
  }
  ss << "StackTrace:\n" << tsl::CurrentStackTrace() << "\n";
  ss << "Status: " << status << "\n";
  XLA_LOG_LINES(tsl::ERROR, ss.str());
  throw std::runtime_error(status.ToString());
}

void CheckComputationStatus(
    const xla::Status& status,
    absl::Span<const xla::XlaComputation* const> computations,
    absl::Span<const xla::Shape* const> output_shapes) {
  if (!status.ok()) {
    ReportComputationError(status, computations, output_shapes);
  }
}

hash_t ShapeHash(const xla::Shape& shape) {
  hash_t hash = 0xa5d2d6916;
  xla::ShapeUtil::ForEachSubshape(
      shape, [&](const xla::Shape& subshape, const xla::ShapeIndex&) {
        hash = SingleShapeHash(subshape, hash);
      });
  return hash;
}

}  // namespace util
}  // namespace runtime
}  // namespace torch_xla
