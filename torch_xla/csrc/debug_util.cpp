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

std::string DebugUtil::GetTensorsGraphInfo(
    tensorflow::gtl::ArraySlice<const XLATensor> tensors,
    const std::vector<size_t>* indices, GraphFormat format) {
  std::stringstream ss;
  std::vector<SourceLocation> frames = GetPythonFrames();
  ss << "TensorsGraphInfo:\n";
  for (auto& location : frames) {
    ss << "  " << location.function << " (" << location.file << ":"
       << location.line << ")\n";
  }
  ss << "\n";

  std::vector<const ir::Node*> roots;
  if (indices != nullptr) {
    for (auto index : *indices) {
      ir::Value ir_value = tensors[index].CurrentIrValue();
      if (ir_value) {
        roots.push_back(ir_value.node.get());
        ss << "Sync tensor with IR: " << roots.back()->ToString() << "\n";
      }
    }
  } else {
    for (auto& tensor : tensors) {
      ir::Value ir_value = tensor.CurrentIrValue();
      if (ir_value) {
        roots.push_back(ir_value.node.get());
        ss << "Sync tensor with IR: " << roots.back()->ToString() << "\n";
      }
    }
  }
  if (format == GraphFormat::kText) {
    ss << "\n" << ir::DumpUtil::ToText(roots) << "\n";
  } else if (format == GraphFormat::kDot) {
    ss << "\n" << ir::DumpUtil::ToText(roots) << "\n";
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
