#include "torch_xla/csrc/debug_util.h"

#include <fstream>
#include <mutex>
#include <sstream>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/ir_util.h"
#include "torch_xla/csrc/python_util.h"

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

}  // namespace

DebugUtil::GraphFormat DebugUtil::GetDefaultGraphFormat() {
  static GraphFormat format = DefaultGraphFormat();
  return format;
}

std::string DebugUtil::GetTensorsGraphInfo(
    tensorflow::gtl::ArraySlice<const XLATensor> tensors,
    const std::vector<size_t>* indices, GraphFormat format) {
  std::vector<const ir::Node*> root_nodes;
  std::vector<ir::Value> root_values;
  if (indices != nullptr) {
    for (auto index : *indices) {
      ir::Value ir_value = tensors[index].CurrentIrValue();
      if (ir_value) {
        root_nodes.push_back(ir_value.node.get());
        root_values.push_back(std::move(ir_value));
      }
    }
  } else {
    for (auto& tensor : tensors) {
      ir::Value ir_value = tensor.CurrentIrValue();
      if (ir_value) {
        root_nodes.push_back(ir_value.node.get());
        root_values.push_back(std::move(ir_value));
      }
    }
  }
  std::stringstream ss;
  std::vector<SourceLocation> frames = GetPythonFrames();
  ss << "TensorsGraphInfo:\n";
  for (auto& location : frames) {
    ss << "  " << location.function << " (" << location.file << ":"
       << location.line << ")\n";
  }
  if (format == GraphFormat::kText) {
    ss << "\n" << ir::DumpUtil::ToText(root_nodes) << "\n";
  } else if (format == GraphFormat::kDot) {
    ss << "\n" << ir::DumpUtil::ToDot(root_nodes) << "\n";
  } else if (format == GraphFormat::kHlo) {
    ss << "\n" << ir::DumpUtil::ToHlo(root_values) << "\n";
  } else {
    XLA_ERROR() << "Invalid graph format: " << format;
  }
  return ss.str();
}

void DebugUtil::SaveTensorsGraphInfo(
    const char* name, tensorflow::gtl::ArraySlice<const XLATensor> tensors,
    const std::vector<size_t>* indices, GraphFormat format) {
  static const std::string save_file =
      xla::sys_util::GetEnvString("XLA_SAVE_TENSORS_FILE", "");
  if (!save_file.empty()) {
    static std::mutex lock;
    std::string info = GetTensorsGraphInfo(tensors, indices, format);
    std::lock_guard<std::mutex> guard(lock);
    std::ofstream graph_file(save_file, std::ios_base::app);
    graph_file << "[" << name << "]\n" << info << "\n";
  }
}

}  // namespace torch_xla
