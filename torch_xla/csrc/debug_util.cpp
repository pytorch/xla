#include "torch_xla/csrc/debug_util.h"

#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/unique.h>
#include <torch/csrc/lazy/python/python_util.h>

#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>
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
#include "torch_xla/csrc/runtime/xla_util.h"
#include "torch_xla/csrc/xla_graph_executor.h"

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
  torch::lazy::Unique<torch::lazy::BackendDevice> unique_device;
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
  torch::lazy::Unique<torch::lazy::BackendDevice> unique_device;
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

  XLAGraphExecutor* graph_executor = XLAGraphExecutor::Get();
  if (graph_executor->CurrentGraphName() != "") {
    ss << "Graph Name: " << graph_executor->CurrentGraphName() << "\n";
  }

  std::vector<torch::lazy::SourceLocation> frames =
      torch::lazy::GetPythonFrames();
  ss << "TensorsGraphInfo:\n";
  for (auto& location : frames) {
    ss << "  " << location.function << " (" << location.file << ":"
       << location.line << ")\n";
  }
  ss << "\nRoot Hashes: (";
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
  ss << "\n## BEGIN_GRAPH\n" << graph_str;
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

void DebugUtil::SaveGraphHash(torch::lazy::hash_t graph_hash) {
  thread_local const std::string save_file =
      runtime::sys_util::GetEnvOrdinalPath(
          "XLA_SAVE_TENSORS_FILE", "", bridge::GetCurrentDevice().ordinal());
  if (!save_file.empty()) {
    // Technically we don't need a lock here as this function should only be
    // called one during each graph execution. Tracing is single thread and
    // blocking. Put a lock here to be save, it is within the debugging tool so
    // perfomrance implcation should be OK.
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    std::ofstream graph_file(save_file, std::ios_base::app);
    graph_file << "Graph Hash: " << torch::lazy::HashToString(graph_hash)
               << "\n\n## END_GRAPH\n\n";
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
      ss << xla::HloSharding::FromProto(xla::HloSharding::Unknown().ToProto())
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

// helper function until we move to C++ 20
static bool endsWith(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
         0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

int GetDebugLevel() {
  static const bool pt_xla_debug_enabled =
      runtime::sys_util::GetEnvBool("PT_XLA_DEBUG", false);
  static const int pt_xla_debug_level_env =
      runtime::sys_util::GetEnvInt("PT_XLA_DEBUG_LEVEL", -1);
  static const int default_debug_level_if_enabled = 100;
  // default the pt_xla_debug_level to 100 if PT_XLA_DEBUG is set but
  // PT_XLA_DEBUG_LEVEL is not specified.
  static const int pt_xla_debug_level =
      (pt_xla_debug_level_env == -1) && pt_xla_debug_enabled
          ? default_debug_level_if_enabled
          : pt_xla_debug_level_env;
  return pt_xla_debug_level;
}

void DebugUtil::analyze_graph_execution_python_frame(
    GraphAnalysisSource source, torch::lazy::hash_t graph_hash,
    const xla::ProgramShape* program_shape) {
  static const int pt_xla_debug_level = GetDebugLevel();
  static const bool is_master_process =
      (runtime::sys_util::GetEnvInt("PJRT_LOCAL_PROCESS_RANK", 0) == 0);
  static const std::string debug_file_name =
      runtime::sys_util::GetEnvString("PT_XLA_DEBUG_FILE", "");
  static const int64_t max_frame_count =
      runtime::sys_util::GetEnvInt("PT_XLA_DEBUG_MAX_FRAME", 8);

  constexpr std::string_view executation_output_prefix = "Execution Analysis: ";
  constexpr std::string_view compilation_output_prefix =
      "Compilation Analysis: ";
  constexpr std::string_view unexpected_execution_prefix =
      "Unexpected Execution Analysis: ";

  XLAGraphExecutor* graph_executor = XLAGraphExecutor::Get();
  bool unexpected_execution = !graph_executor->AllowExecution();

  if (unexpected_execution) {
    // if unexpected_execution happens we want to alywas print
    // debugg message on master process
  } else if (graph_executor->UseEagerMode() &&
             source != GraphAnalysisSource::DynamoExecution) {
    // don't output analysis for eager mode execution/compilation
    return;
  } else if (pt_xla_debug_level <= 0) {
    return;
  } else if (pt_xla_debug_level <= 1 &&
             source != GraphAnalysisSource::Compilation) {
    // for debug level <=1, only output compilation analysis in this function.
    return;
  }

  if (!is_master_process) {
    return;
  }

  std::string_view debug_output_prefix =
      unexpected_execution ? unexpected_execution_prefix
      : (source == GraphAnalysisSource::Compilation)
          ? compilation_output_prefix
          : executation_output_prefix;
  // TODO: Make this configurable.
  std::vector<torch::lazy::SourceLocation> frames =
      torch::lazy::GetPythonFrames();
  // python frame must be > 1
  if (frames.size() == 0) {
    // There is no python frame. Current thread might be started by
    // autograd. Skip the python frame analysis.
    return;
  }
  std::stringstream ss;
  ss << "\n"
     << debug_output_prefix
     << "======================================================================"
        "=========="
     << "\n";
  ss << debug_output_prefix
     << ((source == GraphAnalysisSource::Compilation) ? "Compilation Cause\n"
                                                      : "Execution Cause\n");
  if (source == GraphAnalysisSource::DynamoExecution) {
    // when executation is from dynamo compiled graph, the python stack will not
    // show any dynamo related python file since frame is already replaced. We
    // can either analyze the C++ call stack or rely on caller to pass a boolean
    // variable.
    ss << debug_output_prefix << "  dynamo is executing a compiled program\n";
  } else if (frames[0].function == "mark_step" ||
             (frames[0].function == "sync" &&
              endsWith(frames[0].file, "torch_xla.py"))) {
    if (frames[1].function == "next" &&
        endsWith(frames[1].file, "parallel_loader.py")) {
      ss << debug_output_prefix
         << "  mark_step in parallel loader at step end\n";
    } else if (frames[1].function == "__exit__" &&
               endsWith(frames[1].file, "profiler.py")) {
      ss << debug_output_prefix
         << "  mark_step when exiting a profiler StepTrace region\n";
    } else if ((frames[1].function == "extract_compiled_graph_helper" ||
                frames[1].function == "extract_internal") &&
               endsWith(frames[1].file, "dynamo_bridge.py")) {
      ss << debug_output_prefix
         << "  mark_step when dynamo processing input graphs\n";
    } else if (frames[1].function == "_compile" &&
               endsWith(frames[1].file, "torch_xla.py")) {
      ss << debug_output_prefix << "  torch_xla.compile\n";
    } else if (frames[1].function == "_clear_pending_ops_before_compile" &&
               endsWith(frames[1].file, "torch_xla.py")) {
      ss << debug_output_prefix
         << "  torch_xla.compile clear the pending graph prior calling the "
            "target function\n";
    } else {
      ss << debug_output_prefix << "  user mark_step\n";
    }
  } else if (frames[0].function == "extract_graph_helper" &&
             endsWith(frames[0].file, "dynamo_bridge.py")) {
    ss << debug_output_prefix << "  dynamo is compiling a FX graph to HLO\n";
  } else {
    // TODO(JackCaoG): be more specific about  exeuction caused by printing
    // tensor or fallback or some weird indexing.
    ss << debug_output_prefix
       << "  most likely user code trying to access tensor value before "
          "mark_step\n";
  }

  ss << debug_output_prefix << "Graph Info: \n";
  if (graph_executor->CurrentGraphName() != "") {
    ss << debug_output_prefix
       << "  Graph Name: " << graph_executor->CurrentGraphName() << "\n";
  }
  ss << debug_output_prefix
     << "  Graph Hash: " << torch::lazy::HashToString(graph_hash) << "\n";
  ss << debug_output_prefix
     << "  Number of Graph Inputs: " << program_shape->parameters().size()
     << "\n";
  ss << debug_output_prefix << "  Number of Graph Outputs: "
     << (program_shape->result().IsTuple()
             ? program_shape->result().tuple_shapes_size()
             : 1)
     << "\n";

  int remain_frame_count = max_frame_count;
  ss << debug_output_prefix << "Python Frame Triggered Execution: \n";
  for (auto& location : frames) {
    remain_frame_count--;
    if (remain_frame_count < 0) {
      ss << debug_output_prefix << "  ..........\n";
      break;
    } else {
      ss << debug_output_prefix << "  " << location.function << " ("
         << location.file << ":" << location.line << ")\n";
    }
  }
  ss << debug_output_prefix
     << "----------------------------------------------------------------------"
        "----------"
     << "\n";
  ss << debug_output_prefix
     << "======================================================================"
        "=========="
     << "\n";

  // TODO(JackCaoG): print more information about the graph that is about to get
  // executed.
  if (debug_file_name == "") {
    // print to stderr by default
    std::cerr << ss.str();
  } else {
    std::ofstream outFile;
    outFile.open(debug_file_name, std::ios_base::app);
    outFile << ss.rdbuf();
  }
  if (unexpected_execution) {
    XLA_ERROR() << "Unexpected execution happens inside the compiled function, "
                   "exiting\n";
  }
}

void DebugUtil::post_compilation_analysis(
    runtime::ComputationClient::ComputationPtr computation) {
  static const int pt_xla_debug_level = GetDebugLevel();
  static const bool is_master_process =
      (runtime::sys_util::GetEnvInt("PJRT_LOCAL_PROCESS_RANK", 0) == 0);
  static const std::string debug_file_name =
      runtime::sys_util::GetEnvString("PT_XLA_DEBUG_FILE", "");
  if (pt_xla_debug_level <= 0 || !is_master_process) {
    return;
  }

  // don't output analysis for eager mode execution/compilation.
  // TODO(JackCaoG): enable this for eager+dynamo
  if (XLAGraphExecutor::Get()->UseEagerMode()) {
    return;
  }

  std::stringstream ss;
  // This can be used to verify the hash of the underlying computation proto.
  // Note that for UserComputation computations, the protobuf is factored in
  // the graph hash.
  std::string serialized_computation =
      ConsumeValue(runtime::util::GetDeterministicSerializedModuleProto(
          computation->computation().proto()));
  ss << "\n"
     << "Computation hash: "
     << torch::lazy::HashToString(torch::lazy::Hash(serialized_computation))
     << "\n";

  constexpr std::string_view debug_output_prefix =
      "Post Compilation Analysis: ";
  ss << "\n"
     << debug_output_prefix
     << "======================================================================"
        "=========="
     << "\n";
  std::string memory_info = computation->get_memory_info();

  std::vector<std::string> keysToExtract = {
      "generated_code_size_in_bytes", "argument_size_in_bytes",
      "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes"};
  std::vector<std::string> sizes_in_gb;

  for (const std::string& key : keysToExtract) {
    std::regex pattern(key + "=([0-9]+)");
    std::smatch match;

    if (std::regex_search(memory_info, match, pattern)) {
      sizes_in_gb.push_back(
          std::to_string(std::stoll(match[1]) * 1.0 / 1024 / 1024 / 1024));
    } else {
      sizes_in_gb.push_back("Unknown ");
    }
  }

  ss << debug_output_prefix << "Graph input size: " << sizes_in_gb[1]
     << " GB\n";
  ss << debug_output_prefix << "Graph output size: " << sizes_in_gb[2]
     << " GB\n";
  ss << debug_output_prefix << "Aliased Input size: " << sizes_in_gb[3]
     << " GB\n";
  ss << debug_output_prefix << "Intermediate tensor size: " << sizes_in_gb[4]
     << " GB\n";
  ss << debug_output_prefix << "Compiled program size: " << sizes_in_gb[0]
     << " GB\n";
  ss << debug_output_prefix
     << "----------------------------------------------------------------------"
        "----------"
     << "\n";
  ss << debug_output_prefix
     << "======================================================================"
        "=========="
     << "\n";
  if (debug_file_name == "") {
    // print to stderr by default
    std::cerr << ss.str();
  } else {
    std::ofstream outFile;
    outFile.open(debug_file_name, std::ios_base::app);
    outFile << ss.rdbuf();
  }
}

}  // namespace torch_xla
