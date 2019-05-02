#include "torch_xla/csrc/op_by_op_executor.h"

#include <list>
#include <unordered_map>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
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

absl::optional<size_t> GetOutputIndex(bool is_device_data, size_t index) {
  // The output of every result of an op-by-op computation is wrapped into a
  // tuple, so we need to use the index to extract it. Device data instead is
  // already unwrapped, so we need to pass an empty index so that TF/XRT code
  // uses the result buffer directly.
  if (is_device_data) {
    return absl::nullopt;
  }
  return index;
}

const xla::Shape& GetParameterShape(const ir::Output& operand,
                                    const xla::Shape& input_shape) {
  // See comment in GetOutputIndex() about device data WRT computation outpout
  // shape handling.
  const ir::ops::DeviceData* device_data =
      dynamic_cast<const ir::ops::DeviceData*>(operand.node);
  return device_data != nullptr
             ? input_shape
             : xla::ShapeUtil::GetTupleElementShape(input_shape, operand.index);
}

size_t ComputeNodeKey(
    const ir::Node* node,
    tensorflow::gtl::ArraySlice<const xla::Shape*> input_shapes) {
  size_t key = 0x129b98d6968b7;
  const auto& operands = node->operands();
  for (size_t i = 0; i < operands.size(); ++i) {
    key = xla::util::HashCombine(key, xla::util::ShapeHash(GetParameterShape(
                                          operands[i], *input_shapes[i])));
  }
  key = xla::util::HashCombine(key, xla::util::ShapeHash(node->shape()));
  return xla::util::HashCombine(key, node->node_hash());
}

xla::XlaComputation BuildNodeComputation(
    const ir::Node* node,
    tensorflow::gtl::ArraySlice<const xla::Shape*> input_shapes,
    const Device& device) {
  ir::LoweringContext loctx("BuildNodeComputation");
  const auto& operands = node->operands();
  for (size_t i = 0; i < operands.size(); ++i) {
    xla::XlaOp param = xla::Parameter(
        loctx.builder(), i, GetParameterShape(operands[i], *input_shapes[i]),
        absl::StrCat("param_", i));
    loctx.AssignOutputOp(operands[i], param);
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
  XLA_VALUE_METRIC("OpByOpPostOrderSize", post_order.size());

  std::unordered_map<const ir::Node*, size_t> node_to_index;
  node_to_index.reserve(post_order.size());
  for (size_t i = 0; i < post_order.size(); ++i) {
    node_to_index[post_order[i]] = i;
  }

  Device exec_device(device);
  std::vector<size_t> cache_keys;
  std::unordered_map<size_t, std::vector<size_t>> compile_indices;
  std::unordered_map<size_t, size_t> cache_keys_instance;
  std::list<xla::Shape> compile_shapes;
  std::vector<bool> device_data_ops(post_order.size());
  std::vector<const xla::Shape*> ops_shapes(post_order.size());
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
      ops_shapes[i] = &cxop.device_data->shape();
      device_data_ops[i] = true;
    } else {
      std::vector<const xla::Shape*> op_input_shapes;
      for (auto& operand : node->operands()) {
        size_t op_index = node_to_index.at(operand.node);
        cxop.inputs.push_back(
            {op_index,
             GetOutputIndex(device_data_ops[op_index], operand.index)});
        op_input_shapes.push_back(ops_shapes[op_index]);
      }

      size_t cache_key = ComputeNodeKey(node, op_input_shapes);
      cxop.computation = compile_cache_.Get(cache_key);
      if (cxop.computation == nullptr) {
        XLA_COUNTER("OpByOpCompileCacheMiss", 1);

        // Within a single IR graph, there can be many duplicated IR nodes, so
        // make sure we do not issue an XLA compilation for each one of those.
        auto& cache_key_indices = compile_indices[cache_key];
        cache_key_indices.push_back(i);
        if (cache_key_indices.size() == 1) {
          cache_keys.push_back(cache_key);
          cache_keys_instance[cache_key] = compile_instances.size();

          xla::XlaComputation computation =
              BuildNodeComputation(node, op_input_shapes, exec_device);
          xla::ProgramShape program_shape =
              ConsumeValue(computation.GetProgramShape());
          compile_shapes.push_back(MakeShapeWithDeviceLayout(
              program_shape.result(), exec_device.hw_type));
          compile_instances.push_back(
              {std::move(computation),
               xla::ComputationClient::Get()->GetCompilationDevices(device),
               &compile_shapes.back()});

          ops_shapes[i] = &compile_shapes.back();
        } else {
          ops_shapes[i] =
              compile_instances[cache_keys_instance.at(cache_key)].output_shape;
        }
      } else {
        ops_shapes[i] = &cxop.computation->program_shape().result();
      }
    }
  }
  // Fixup the requested outputs (roots) within the chained ops vector.
  for (size_t i = 0; i < roots.size(); ++i) {
    size_t op_index = node_to_index.at(roots[i].node.get());
    chained_exec_ops[op_index].outputs.push_back(
        {i, GetOutputIndex(device_data_ops[op_index], roots[i].index)});
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
