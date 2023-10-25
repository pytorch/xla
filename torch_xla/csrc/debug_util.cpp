#include "torch_xla/csrc/debug_util.h"

#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/python/python_util.h>

#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_set>

#include "absl/memory/memory.h"
#include "absl/strings/str_split.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/runtime/unique.h"
#include "torch_xla/csrc/xla_graph_executor.h"

#include <iostream>
using std::cerr;

namespace torch_xla {
namespace {

DebugUtil::GraphFormat DefaultGraphFormat() {
  std::string fmt_str =
      runtime::sys_util::GetEnvString("XLA_SAVE_TENSORS_FMT", "text");
  if (fmt_str == "text") {
    return DebugUtil::GraphFormat::kText;
  } else if (fmt_str == "hlo") {
    return DebugUtil::GraphFormat::kHlo;
  } else if (fmt_str == "dot") {
    return DebugUtil::GraphFormat::kDot;
  } else if (fmt_str == "stablehlo") {
    return DebugUtil::GraphFormat::kStableHlo;
  }
  XLA_ERROR() << "Invalid save graph format: " << fmt_str;
}

std::unordered_set<std::string>* LoadExperiments() {
  std::unique_ptr<std::unordered_set<std::string>> xset =
      absl::make_unique<std::unordered_set<std::string>>();
  std::string experiments =
      runtime::sys_util::GetEnvString("XLA_EXPERIMENTAL", "");
  std::vector<std::string> experiment_list = absl::StrSplit(experiments, ':');
  for (auto& name : experiment_list) {
    xset->insert(name);
  }
  return xset.release();
}

}  // namespace

DebugUtil::GraphFormat DebugUtil::GetDefaultGraphFormat() {
  static GraphFormat format = DefaultGraphFormat();
  return format;
}

std::string DebugUtil::GetTensorsGraphHlo(
    absl::Span<const XLATensorPtr> tensors, const std::vector<size_t>* indices,
    bool dump_stablehlo) {
  std::vector<torch::lazy::Value> root_values;
  runtime::util::Unique<torch::lazy::BackendDevice> unique_device;
  if (indices != nullptr) {
    for (auto index : *indices) {
      const XLATensorPtr& tensor = tensors[index];
      torch::lazy::Value ir_value = tensor->CurrentIrValue();
      if (ir_value) {
        root_values.push_back(std::move(ir_value));
        unique_device.set(tensor->GetDevice());
      }
    }
  } else {
    for (auto& tensor : tensors) {
      torch::lazy::Value ir_value = tensor->CurrentIrValue();
      if (ir_value) {
        root_values.push_back(std::move(ir_value));
        unique_device.set(tensor->GetDevice());
      }
    }
  }
  return DumpUtil::ToHlo(
      root_values, unique_device ? *unique_device : bridge::GetCurrentDevice(),
      EmitMode::kStableHloReadable);
}

std::string DebugUtil::GetTensorsGraphInfo(
    absl::Span<const XLATensorPtr> tensors, const std::vector<size_t>* indices,
    GraphFormat format) {
  std::vector<const torch::lazy::Node*> root_nodes;
  std::vector<torch::lazy::Value> root_values;
  std::vector<torch::lazy::hash_t> root_hashes;
  runtime::util::Unique<torch::lazy::BackendDevice> unique_device;
  if (indices != nullptr) {
    for (auto index : *indices) {
      const XLATensorPtr& tensor = tensors[index];
      torch::lazy::Value ir_value = tensor->CurrentIrValue();
      if (ir_value) {
        root_nodes.push_back(ir_value.node.get());
        root_hashes.push_back(ir_value.hash());
        root_values.push_back(std::move(ir_value));
        unique_device.set(tensor->GetDevice());
      }
    }
  } else {
    for (auto& tensor : tensors) {
      torch::lazy::Value ir_value = tensor->CurrentIrValue();
      if (ir_value) {
        root_nodes.push_back(ir_value.node.get());
        root_hashes.push_back(ir_value.hash());
        root_values.push_back(std::move(ir_value));
        unique_device.set(tensor->GetDevice());
      }
    }
  }
  std::stringstream ss;
  std::vector<torch::lazy::SourceLocation> frames =
      torch::lazy::GetPythonFrames();
  ss << "TensorsGraphInfo:\n";
  for (auto& location : frames) {
    ss << "  " << location.function << " (" << location.file << ":"
       << location.line << ")\n";
  }
  ss << "\nHashes: (";
  for (size_t i = 0; i < root_hashes.size(); ++i) {
    if (i > 0) {
      ss << ", ";
    }
    ss << torch::lazy::HashToString(root_hashes[i]);
  }
  ss << ")\n";

  std::string graph_str;
  if (format == GraphFormat::kText) {
    graph_str = DumpUtil::ToText(root_nodes);
  } else if (format == GraphFormat::kDot) {
    graph_str = DumpUtil::ToDot(root_nodes);
  } else if (format == GraphFormat::kHlo) {
    graph_str = DumpUtil::ToHlo(root_values, unique_device
                                                 ? *unique_device
                                                 : bridge::GetCurrentDevice());
  } else if (format == GraphFormat::kStableHlo) {
    graph_str = DumpUtil::ToHlo(
        root_values,
        unique_device ? *unique_device : bridge::GetCurrentDevice(),
        EmitMode::kStableHloReadable);
  } else {
    XLA_ERROR() << "Invalid graph format: " << format;
  }
  ss << "\n## BEGIN_GRAPH\n" << graph_str << "\n## END_GRAPH\n\n";
  return ss.str();
}

void DebugUtil::SaveTensorsGraphInfo(const char* name,
                                     absl::Span<const XLATensorPtr> tensors,
                                     const std::vector<size_t>* indices,
                                     GraphFormat format) {
  thread_local const std::string save_file =
      runtime::sys_util::GetEnvOrdinalPath(
          "XLA_SAVE_TENSORS_FILE", "", bridge::GetCurrentDevice().ordinal());
  if (!save_file.empty()) {
    static std::mutex lock;
    if ((format == DebugUtil::GraphFormat::kHlo ||
         format == DebugUtil::GraphFormat::kStableHlo) &&
        indices->size() > 0) {
      // Dumping the HLO might access the placeholder data created during
      // previous execution. We need to wait for last execution to finish before
      // proceeding.
      torch::lazy::BackendDevice device = tensors[(*indices)[0]]->GetDevice();
      XLAGraphExecutor::Get()->WaitDeviceOps({device.toString()});
    }
    std::string info = GetTensorsGraphInfo(tensors, indices, format);
    std::lock_guard<std::mutex> guard(lock);
    std::ofstream graph_file(save_file, std::ios_base::app);
    graph_file << "[" << name << "]\n" << info << "\n";
  }
}

void DebugUtil::SaveOutputShardingInfo(std::vector<XLATensorPtr>* tensors,
                                       absl::Span<const size_t> indices) {
  thread_local const std::string save_file =
      runtime::sys_util::GetEnvOrdinalPath(
          "XLA_SAVE_TENSORS_FILE", "", bridge::GetCurrentDevice().ordinal());
  std::string fmt_str =
      runtime::sys_util::GetEnvString("XLA_SAVE_TENSORS_FMT", "text");
  if (save_file.empty() || fmt_str != "hlo") {
    return;
  }
  std::stringstream ss;
  for (int i = 0; i < indices.size(); ++i) {
    auto xtensor = (*tensors)[indices[i]];
    ss << xtensor->shape().get().ToString() << " ";
    if (xtensor->sharding_spec()) {
      ss << xla::HloSharding::FromProto(xtensor->sharding_spec()->sharding)
                ->ToString();
    } else {
      ss << xla::HloSharding::FromProto(xla::HloSharding::Replicate().ToProto())
                ->ToString();
    }
    ss << "\n";
  }
  std::ofstream graph_file(save_file, std::ios_base::app);
  graph_file << "\n#OUTPUT_SHARDING_BEGIN\n\n"
             << ss.str() << "\n#OUTPUT_SHARDING_END\n\n";
}

bool DebugUtil::ExperimentEnabled(const std::string& name) {
  static const std::unordered_set<std::string>* xset = LoadExperiments();
  return xset->find(name) != xset->end();
}

void DebugUtil::analyze_graph_execution_python_frame() {
  // TODO: maybe only print this for the master process
  static std::string debug_output_prefix = "Execution Analysis: ";
  std::vector<torch::lazy::SourceLocation> frames =
      torch::lazy::GetPythonFrames();
  // python frame must be > 1
  XLA_CHECK_GE(frames.size(), 1);
  std::stringstream ss;
  ss << debug_output_prefix
     << "======================================================================"
        "=========="
     << "\n";
  if (frames[0].function == "mark_step") {
    // TODO: be more specified about the caller of the mark step
    // for example: parallelr loader, dynamo, step_trace, user code etc
    ss << debug_output_prefix << "execution is caused by mark_step\n";
  }
  // handle the exeuction caused by printing tensor or fallback or some
  // weird indexing.

  // make number of frames printed configurable
  ss << debug_output_prefix << "Python Frame triggered execution: \n";
  for (auto& location : frames) {
    ss << debug_output_prefix << "  " << location.function << " ("
       << location.file << ":" << location.line << ")\n";
  }
  ss << debug_output_prefix
     << "----------------------------------------------------------------------"
        "----------"
     << "\n";
  ss << debug_output_prefix
     << "======================================================================"
        "=========="
     << "\n";

  // print more information about the graph that is about to get executed.
  cerr << ss.str();
}

}  // namespace torch_xla
