#include "torch_xla/csrc/debug_util.h"

#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/python/python_util.h>

#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_set>

#include "absl/memory/memory.h"
#include "absl/strings/str_split.h"
#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/sys_util.h"
#include "third_party/xla_client/unique.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/ir_util.h"
#include "torch_xla/csrc/xla_graph_executor.h"

namespace torch_xla {
namespace {

DebugUtil::GraphFormat DefaultGraphFormat() {
  std::string fmt_str =
      xla::sys_util::GetEnvString("XLA_SAVE_TENSORS_FMT", "text");
  if (fmt_str == "text") {
    return DebugUtil::GraphFormat::kText;
  } else if (fmt_str == "hlo") {
    return DebugUtil::GraphFormat::kHlo;
  } else if (fmt_str == "dot") {
    return DebugUtil::GraphFormat::kDot;
  }
  XLA_ERROR() << "Invalid save graph format: " << fmt_str;
}

std::unordered_set<std::string>* LoadExperiments() {
  std::unique_ptr<std::unordered_set<std::string>> xset =
      absl::make_unique<std::unordered_set<std::string>>();
  std::string experiments = xla::sys_util::GetEnvString("XLA_EXPERIMENTAL", "");
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

std::string DebugUtil::GetTensorsGraphInfo(
    absl::Span<const XLATensorPtr> tensors, const std::vector<size_t>* indices,
    GraphFormat format) {
  std::vector<const torch::lazy::Node*> root_nodes;
  std::vector<torch::lazy::Value> root_values;
  std::vector<torch::lazy::hash_t> root_hashes;
  xla::util::Unique<torch::lazy::BackendDevice> unique_device;
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
    graph_str = DumpUtil::ToHlo(
        root_values, unique_device ? *unique_device : GetCurrentDevice());
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
  thread_local const std::string save_file = xla::sys_util::GetEnvOrdinalPath(
      "XLA_SAVE_TENSORS_FILE", "", GetCurrentDevice().ordinal());
  if (!save_file.empty()) {
    static std::mutex lock;
    if (format == DebugUtil::GraphFormat::kHlo && indices->size() > 0) {
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

bool DebugUtil::ExperimentEnabled(const std::string& name) {
  static const std::unordered_set<std::string>* xset = LoadExperiments();
  return xset->find(name) != xset->end();
}

}  // namespace torch_xla
