#include "torch_xla/csrc/op_by_op_executor.h"

#include <list>
#include <unordered_map>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

size_t ComputeNodeKey(const ir::Node* node) {
  size_t key = 0x129b98d6968b7;
  for (auto& operand : node->operands()) {
    key =
        xla::util::HashCombine(key, xla::util::ShapeHash(operand.node_shape()));
  }
  key = xla::util::HashCombine(key, xla::util::ShapeHash(node->shape()));
  return xla::util::HashCombine(key, node->node_hash());
}

xla::XlaComputation BuildNodeComputation(const ir::Node* node,
                                         const Device& device) {
  ir::LoweringContext loctx("BuildNodeComputation");
  size_t param_no = 0;
  for (auto& operand : node->operands()) {
    xla::Shape parameter_shape =
        MakeShapeWithDeviceLayout(operand.shape(), device.hw_type);
    xla::XlaOp param =
        xla::Parameter(loctx.builder(), param_no, parameter_shape,
                       absl::StrCat("param_", param_no));
    ++param_no;
    loctx.AssignOutputOp(operand, param);
  }
  for (auto& xla_op : loctx.LowerNode(node)) {
    loctx.AddResult(xla_op);
  }
  return ConsumeValue(loctx.Build());
}

}  // namespace

OpByOpExecutor::OpByOpExecutor(size_t compile_cache_size)
    : compile_cache_(compile_cache_size) {}

std::vector<xla::ComputationClient::ExecuteChainedOp> OpByOpExecutor::BuildOps(
    tensorflow::gtl::ArraySlice<const ir::Value> roots,
    const std::string& device) {
  std::vector<const ir::Node*> root_nodes;
  root_nodes.reserve(roots.size());
  for (auto& root : roots) {
    root_nodes.push_back(root.node.get());
  }
  std::vector<const ir::Node*> post_order =
      ir::Util::ComputePostOrder(root_nodes);

  std::unordered_map<const ir::Node*, size_t> node_to_index;
  node_to_index.reserve(post_order.size());
  for (size_t i = 0; i < post_order.size(); ++i) {
    node_to_index[post_order[i]] = i;
  }

  Device exec_device(device);
  std::vector<size_t> cache_keys;
  std::unordered_map<size_t, std::vector<size_t>> compile_indices;
  std::list<xla::Shape> compile_shapes;
  std::vector<xla::ComputationClient::CompileInstance> compile_instances;
  std::vector<xla::ComputationClient::ExecuteChainedOp> chained_exec_ops(
      post_order.size());
  for (size_t i = 0; i < post_order.size(); ++i) {
    const ir::Node* node = post_order[i];
    xla::ComputationClient::ExecuteChainedOp& cxop = chained_exec_ops[i];
    const ir::ops::DeviceData* device_data =
        dynamic_cast<const ir::ops::DeviceData*>(node);
    if (device_data != nullptr) {
      cxop.device_data = device_data->data();
    } else {
      size_t cache_key = ComputeNodeKey(node);
      cxop.computation = compile_cache_.Get(cache_key);
      if (cxop.computation == nullptr) {
        XLA_COUNTER("OpByOpCompileCacheMiss", 1);

        // Within a single IR graph, there can be many duplicated IR nodes, so
        // make sure we do not issue an XLA compilation for each one of those.
        auto& cache_key_indices = compile_indices[cache_key];
        cache_key_indices.push_back(i);
        if (cache_key_indices.size() == 1) {
          cache_keys.push_back(cache_key);

          xla::XlaComputation computation =
              BuildNodeComputation(node, exec_device);
          xla::ProgramShape program_shape =
              ConsumeValue(computation.GetProgramShape());
          compile_shapes.push_back(MakeShapeWithDeviceLayout(
              program_shape.result(), exec_device.hw_type));
          compile_instances.push_back(
              {std::move(computation),
               xla::ComputationClient::Get()->GetCompilationDevices(device),
               &compile_shapes.back()});
        }
      }
      for (auto& operand : node->operands()) {
        cxop.inputs.push_back({node_to_index.at(operand.node), operand.index});
      }
    }
  }
  // Fixup the requested outputs (roots) within the chained ops vector.
  for (size_t i = 0; i < roots.size(); ++i) {
    size_t op_index = node_to_index.at(roots[i].node.get());
    chained_exec_ops[op_index].outputs.push_back({roots[i].index, i});
  }
  // If we missed the cache for certain ops, compile them now and fixup the
  // chained ops vector.
  if (!compile_instances.empty()) {
    auto computation_ptrs =
        xla::ComputationClient::Get()->Compile(std::move(compile_instances));
    for (size_t i = 0; i < computation_ptrs.size(); ++i) {
      compile_cache_.Add(cache_keys[i], computation_ptrs[i]);
      for (auto index : compile_indices[cache_keys[i]]) {
        chained_exec_ops[index].computation = computation_ptrs[i];
      }
    }
  }
  return chained_exec_ops;
}

std::vector<xla::ComputationClient::DataPtr> OpByOpExecutor::Execute(
    tensorflow::gtl::ArraySlice<const ir::Value> roots,
    const std::string& device) {
  auto chained_exec_ops = BuildOps(roots, device);
  return xla::ComputationClient::Get()->ExecuteChained(chained_exec_ops,
                                                       device);
}

OpByOpExecutor::AsyncTask OpByOpExecutor::ExecuteAsync(
    tensorflow::gtl::ArraySlice<const ir::Value> roots,
    const std::string& device) {
  std::vector<ir::Value> roots_vector(roots.begin(), roots.end());
  auto taskfn = [this, roots = std::move(roots_vector),
                 device]() -> AsyncResult { return Execute(roots, device); };

  AsyncTask async = AsyncTask(std::move(taskfn));
  return async.Schedule();
}

OpByOpExecutor* OpByOpExecutor::Get() {
  static const xla::int64 compile_cache_size =
      xla::sys_util::GetEnvInt("SPLIT_EXECUTOR_CACHE_SIZE", 2048);
  static OpByOpExecutor* split_executor =
      new OpByOpExecutor(compile_cache_size);
  return split_executor;
}

}  // namespace torch_xla
