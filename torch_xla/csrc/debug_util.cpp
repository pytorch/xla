#include "torch_xla/csrc/debug_util.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/ir_util.h"
#include "torch_xla/csrc/python_util.h"

namespace torch_xla {

std::string DebugUtil::GetTensorsGraphInfo(
    const std::vector<XLATensor>& tensors, const std::vector<size_t>* indices,
    GraphFormat format) {
  std::vector<const ir::Node*> roots;
  if (indices != nullptr) {
    for (auto index : *indices) {
      ir::Value ir_value = tensors[index].CurrentIrValue();
      if (ir_value) {
        roots.push_back(ir_value.node.get());
      }
    }
  } else {
    for (auto& tensor : tensors) {
      ir::Value ir_value = tensor.CurrentIrValue();
      if (ir_value) {
        roots.push_back(ir_value.node.get());
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
    ss << "\n" << ir::DumpUtil::ToText(roots) << "\n";
  } else if (format == GraphFormat::kDot) {
    ss << "\n" << ir::DumpUtil::ToText(roots) << "\n";
  } else {
    XLA_ERROR() << "Invalid graph format: " << format;
  }
  return ss.str();
}

}  // namespace torch_xla
