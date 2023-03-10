#include "torch_xla/csrc/xla_graph_executor.h"

#include <Python.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/util.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"
#include "third_party/xla_client/cache.h"
#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/env_vars.h"
#include "third_party/xla_client/sys_util.h"
#include "third_party/xla_client/thread_pool.h"
#include "third_party/xla_client/unique.h"
#include "third_party/xla_client/xla_util.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/computation.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/layout_manager.h"
#include "torch_xla/csrc/op_by_op_executor.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/cast.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/ops/dynamic_ir.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/expand_symint.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/ops/view.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/xla_backend_impl.h"
#include "torch_xla/csrc/xla_sharding_util.h"

namespace torch_xla {
namespace {

torch::lazy::Value IrValueFromScalar(const at::Scalar& value,
                                     at::ScalarType scalar_type,
                                     const torch::lazy::BackendDevice& device) {
  at::Tensor tensor = at::scalar_tensor(value, at::TensorOptions(scalar_type));
  torch::lazy::BackendDataPtr device_data = TensorToXlaData(tensor, device);
  return torch::lazy::MakeNode<DeviceData>(std::move(device_data));
}

bool ShouldSyncIrValue(const torch::lazy::Value& ir_value) {
  return ir_value->op() != xla_not_supported;
}

}  // namespace

auto XLAGraphExecutor::DeviceContextArena::Get() -> DeviceContextArena* {
  static DeviceContextArena* arena = new DeviceContextArena();
  return arena;
}

std::vector<XLATensorPtr> XLAGraphExecutor::DeviceContextArena::GetLiveTensors(
    const torch::lazy::BackendDevice* device) {
  std::vector<XLATensorPtr> tensors;
  auto fn = [&](DeviceContext* devctx) {
    std::lock_guard<std::mutex> lock(devctx->lock);
    for (auto& uid_wptr : devctx->tensors_data) {
      auto data =
          std::dynamic_pointer_cast<XLATensor::Data>(uid_wptr.second.lock());
      if (data != nullptr) {
        tensors.push_back(XLATensor::Create(std::move(data)));
      }
    }
  };
  ForAllDeviceContexts(fn, device);
  return tensors;
}

torch::lazy::Value XLAGraphExecutor::DeviceContextArena::GetRngSeed(
    const torch::lazy::BackendDevice& device) {
  static const at::ScalarType kSeedType = at::ScalarType::Long;
  static const uint64_t kSeedMul = 214013;
  static const uint64_t kSeedAdd = 2531011;
  DeviceContext* devctx = GetDeviceContext(device);
  std::lock_guard<std::mutex> lock(devctx->lock);
  if (!devctx->seed_ir_value) {
    devctx->seed_ir_value =
        IrValueFromScalar(MakeIntScalar(devctx->seed), kSeedType, device);
  }
  // Keep the running seed as scalar as well, so we can return it directly
  // without executing graphs.
  devctx->running_seed = kSeedAdd + kSeedMul * devctx->running_seed;
  // Compose new seeds from the root seed, to avoid creating too many XLA
  // computation parameters which might overflow the TPU capacity.
  torch::lazy::Value k = ScalarOp(MakeIntScalar(kSeedMul),
                                  MakeXlaPrimitiveType(kSeedType, &device));
  torch::lazy::Value b = ScalarOp(MakeIntScalar(kSeedAdd),
                                  MakeXlaPrimitiveType(kSeedType, &device));
  devctx->seed_ir_value = b + k * devctx->seed_ir_value;
  return devctx->seed_ir_value;
}

torch::lazy::BackendDataPtr
XLAGraphExecutor::DeviceContextArena::GetBaseSeedData(
    const torch::lazy::BackendDevice& device) {
  static const at::ScalarType kSeedType = at::ScalarType::Long;
  DeviceContext* devctx = GetDeviceContext(device);
  std::lock_guard<std::mutex> lock(devctx->lock);
  at::Tensor tensor = at::scalar_tensor(MakeIntScalar(devctx->seed),
                                        at::TensorOptions(kSeedType));
  torch::lazy::BackendDataPtr device_data = TensorToXlaData(tensor, device);
  devctx->seed_ir_value = torch::lazy::MakeNode<DeviceData>(device_data);
  devctx->running_seed = devctx->seed;
  return torch_xla::DeviceData::Cast(devctx->seed_ir_value.node.get())->data();
}

void XLAGraphExecutor::DeviceContextArena::SaveGraphAsString(
    torch::lazy::hash_t hash, absl::Span<const XLATensorPtr> tensors,
    const std::vector<size_t>* indices, DebugUtil::GraphFormat format) {
  static bool should_save_graph =
      xla::sys_util::GetEnvOrdinalPath("XLA_SAVE_TENSORS_FILE", "",
                                       GetCurrentDevice().ordinal()) != "";
  if (should_save_graph &&
      hash_to_graph_map.find(hash) == hash_to_graph_map.end()) {
    hash_to_graph_map[hash] =
        DebugUtil::GetTensorsGraphInfo(tensors, indices, format);
  }
}

void XLAGraphExecutor::DeviceContextArena::SaveOutputShapes(
    torch::lazy::hash_t hash, std::vector<xla::Shape> output_shapes) {
  if (hash_to_output_shape_map.find(hash) == hash_to_output_shape_map.end()) {
    hash_to_output_shape_map[hash] = std::move(output_shapes);
  }
}

std::string XLAGraphExecutor::DeviceContextArena::GetGraphByHash(
    torch::lazy::hash_t hash) {
  auto iter = hash_to_graph_map.find(hash);
  if (iter == hash_to_graph_map.end()) {
    TF_LOG(INFO) << "Trying to dump graph with an invalid hash";
    return "";
  }
  return iter->second;
}

std::vector<xla::Shape>*
XLAGraphExecutor::DeviceContextArena::GetOutputShapesByHash(
    torch::lazy::hash_t hash) {
  auto iter = hash_to_output_shape_map.find(hash);
  XLA_CHECK(iter != hash_to_output_shape_map.end())
      << "Hash not found, can't retrive output shape";
  return &(iter->second);
}

torch::lazy::Value XLAGraphExecutor::DeviceContextArena::IrValueFromScalar(
    const at::Scalar& value, at::ScalarType scalar_type,
    const torch::lazy::BackendDevice& device) {
  at::Tensor tensor = at::scalar_tensor(value, at::TensorOptions(scalar_type));
  torch::lazy::BackendDataPtr device_data = TensorToXlaData(tensor, device);
  return torch::lazy::MakeNode<DeviceData>(std::move(device_data));
}

XLAGraphExecutor::Async::Async(
    SyncTensorCollection* coll,
    std::vector<torch::lazy::BackendDataPtr> parameters_data,
    std::vector<torch::lazy::BackendDataPtr> tensors_data,
    ComputationCache::TypePtr cached_computation)
    : torch::lazy::LazyGraphExecutor::Async(coll, parameters_data, tensors_data,
                                            nullptr),
      cached_computation(std::move(cached_computation)) {}

XLAGraphExecutor* XLAGraphExecutor::Get() {
  static XLAGraphExecutor arena = XLAGraphExecutor();
  return &arena;
}

void XLAGraphExecutor::RegisterTensor(
    std::shared_ptr<torch::lazy::LazyTensor::Data> data) {
  DeviceContextArena::Get()->RegisterTensor(data);
  TORCH_LAZY_COUNTER("CreateXlaTensor", 1);
}

void XLAGraphExecutor::UnregisterTensor(torch::lazy::LazyTensor::Data* data) {
  DeviceContextArena::Get()->UnregisterTensor(data);
  TORCH_LAZY_COUNTER("DestroyXlaTensor", 1);
}

void XLAGraphExecutor::ApplyEagerSync(std::vector<XLATensorPtr>& tensors) {
  SyncTensorsGraph(&tensors, {}, /*wait=*/false, /*sync_ltc_data=*/false);
}

torch::lazy::Value XLAGraphExecutor::GetDeviceDataIrValue(
    const at::Scalar& value, xla::PrimitiveType type,
    const torch::lazy::BackendDevice& device) {
  torch::lazy::BackendDataPtr data =
      GetDeviceData(value, TensorTypeFromXlaType(type), device);
  data->SetInfo(
      std::make_shared<torch::lazy::LazyGraphExecutor::DeviceDataInfo>(
          /*tensor_id=*/-1, /*read_only=*/true));
  return torch::lazy::MakeNode<DeviceData>(std::move(data));
}

torch::lazy::Value XLAGraphExecutor::GetIrValueForConstant(
    const at::Scalar& value, const xla::Shape& shape) {
  torch::lazy::Value ir_value =
      ScalarOp(std::move(value), shape.element_type());
  if (!shape.dimensions().empty()) {
    ir_value = torch::lazy::MakeNode<Expand>(
        ir_value, torch::lazy::ToVector<int64_t>(shape.dimensions()));
  }
  return ir_value;
}

torch::lazy::Value XLAGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value, xla::PrimitiveType type,
    const torch::lazy::BackendDevice& device) {
  if (torch::lazy::IsSpecialScalar(value)) {
    return ScalarOp(std::move(value), type);
  }
  return GetDeviceDataIrValue(value, type, device);
}

torch::lazy::Value XLAGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value, const torch::lazy::BackendDevice& device) {
  return GetIrValueForScalar(
      value, MakeXlaPrimitiveType(GetScalarType(value), &device), device);
}

torch::lazy::Value XLAGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value, xla::PrimitiveType type,
    absl::Span<const int64_t> dimensions,
    const torch::lazy::BackendDevice& device) {
  torch::lazy::Value ir_value = GetIrValueForScalar(value, type, device);
  if (!dimensions.empty()) {
    ir_value = torch::lazy::MakeNode<Expand>(
        ir_value, torch::lazy::ToVector<int64_t>(dimensions));
  }
  return ir_value;
}

torch::lazy::Value XLAGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value, xla::PrimitiveType type,
    c10::SymIntArrayRef sym_size, const torch::lazy::BackendDevice& device) {
  torch::lazy::Value ir_value = GetIrValueForScalar(value, type, device);
  SymIntElements size_elements = SymIntElements(sym_size);
  return torch::lazy::MakeNode<ExpandSymInt>(ir_value, size_elements);
}

torch::lazy::Value XLAGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value, const xla::Shape& shape,
    const torch::lazy::BackendDevice& device) {
  return GetIrValueForScalar(value, shape.element_type(), shape.dimensions(),
                             device);
}

torch::lazy::Value XLAGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value, const xla::Shape& shape,
    c10::optional<at::ScalarType> logical_element_type,
    const torch::lazy::BackendDevice& device) {
  xla::PrimitiveType type =
      logical_element_type
          ? MakeXlaPrimitiveType(*logical_element_type, &device)
          : shape.element_type();
  return GetIrValueForScalar(value, type, shape.dimensions(), device);
}

torch::lazy::Value XLAGraphExecutor::GetRngSeed(
    const torch::lazy::BackendDevice& device) {
  return DeviceContextArena::Get()->GetRngSeed(device);
}

void XLAGraphExecutor::SetRngSeed(const torch::lazy::BackendDevice& device,
                                  uint64_t seed) {
  DeviceContextArena::Get()->SetRngSeed(device, seed);
}

uint64_t XLAGraphExecutor::GetRunningSeed(
    const torch::lazy::BackendDevice& device) {
  return DeviceContextArena::Get()->GetRunningSeed(device);
}

torch::lazy::BackendDataPtr XLAGraphExecutor::GetBaseSeedData(
    const torch::lazy::BackendDevice& device) {
  return DeviceContextArena::Get()->GetBaseSeedData(device);
}

std::string XLAGraphExecutor::DumpHloComputation(
    const std::vector<XLATensorPtr>& tensors) {
  std::vector<torch::lazy::Value> ir_values;
  for (auto& tensor : tensors) {
    torch::lazy::Value ir_value = tensor->CurrentIrValue();
    if (ir_value) {
      ir_values.push_back(std::move(ir_value));
    }
  }
  return !ir_values.empty() ? DumpUtil::ToHlo(ir_values, GetCurrentDevice())
                            : std::string();
}

std::vector<XLATensorPtr> XLAGraphExecutor::GetLiveTensors(
    const torch::lazy::BackendDevice* device) {
  return DeviceContextArena::Get()->GetLiveTensors(device);
}

void XLAGraphExecutor::SyncTensorsGraph(std::vector<XLATensorPtr>* tensors,
                                        absl::Span<const std::string> devices,
                                        bool wait, bool sync_ltc_data,
                                        bool warm_up_cache_only) {
  TF_VLOG(4) << "Trying to sync the value of " << tensors->size()
             << " tensor(s)";
  tsl::profiler::TraceMe activity("SyncTensorsGraph",
                                  tsl::profiler::TraceMeLevel::kInfo);
  static const bool op_by_op =
      xla::sys_util::GetEnvBool("XLA_SYNC_TENSORS_OPBYOP", false);
  SyncTensorsConfig config;
  config.sync_ltc_data = sync_ltc_data;
  if (warm_up_cache_only) {
    config.force_ltc_data = false;
  }
  if (op_by_op) {
    OpByOpAsync async = SyncTensorsGraphOpByOp(tensors, devices, config);
    if (wait) {
      async.Wait();
    }
  } else {
    auto async =
        SyncTensorsGraphInternal(tensors, devices, config, warm_up_cache_only);
    if (wait && async != nullptr && !warm_up_cache_only) {
      async->mwait.Wait();
    }
  }
}

void XLAGraphExecutor::SyncLiveTensorsGraph(
    const torch::lazy::BackendDevice* device,
    c10::ArrayRef<std::string> devices, bool wait) {
  tsl::profiler::TraceMe activity("SyncLiveTensorsGraph",
                                  tsl::profiler::TraceMeLevel::kInfo);
  auto tensors = GetLiveTensors(device);
  TF_VLOG(4) << tensors.size() << " live tensors: devices=("
             << c10::Join(",", devices) << ")";
  SyncTensorsGraph(&tensors, devices, wait, /*sync_ltc_data=*/true);
}

void XLAGraphExecutor::MarkStep(const torch::lazy::BackendDevice& device) {
  // TODO(jwtan): Replace this with TORCH_LAZY_COUNTER. We need MarkStep to
  // remain as XLA_COUNTER to support xla::metrics::CreatePerformanceReport().
  // For more information, see NOTE: [TORCH_LAZY_COUNTER v.s. XLA_COUNTER].
  XLA_COUNTER("MarkStep", 1);
  DeviceContextArena::Get()->MarkStep(device);
  torch::lazy::ScopePusher::ResetScopes();
  ResetTrimCounter();
}

void XLAGraphExecutor::WaitDeviceOps(absl::Span<const std::string> devices) {
  std::set<torch::lazy::BackendDevice> wait_devices;
  if (!devices.empty()) {
    for (auto& device_str : devices) {
      wait_devices.insert(ParseDeviceString(device_str));
    }
  } else {
    for (auto& device_str : xla::ComputationClient::Get()->GetLocalDevices()) {
      wait_devices.insert(ParseDeviceString(device_str));
    }
  }
  // The DeviceLockerArena::Get()->LockDevices() API returns a vector of
  // torch::lazy::ExceptionCleanup object, which is going to be freed
  // immediately, turning this operation into a lock barrier.
  DeviceLockerArena::Get()->LockDevices(wait_devices);
}

std::vector<at::Tensor> XLAGraphExecutor::GetTensors(
    std::vector<XLATensorPtr>* tensors) {
  TF_VLOG(4) << "Trying to get the value of " << tensors->size()
             << " tensor(s)";
  static const bool op_by_op =
      xla::sys_util::GetEnvBool("XLA_GET_TENSORS_OPBYOP", false);
  return op_by_op ? GetTensorsOpByOp(tensors) : GetTensorsFused(tensors);
}

torch::lazy::hash_t XLAGraphExecutor::GetGraphHash(
    const std::vector<XLATensorPtr>& tensors) {
  SyncTensorsConfig config;
  config.sync_ltc_data = true;
  config.force_ltc_data = false;

  SyncTensorCollection coll = CollectSyncTensors(tensors, config);
  absl::Span<const size_t> indices = coll.indices;
  std::vector<torch::lazy::Value> ir_values;
  std::vector<xla::Shape> output_shapes;
  ir_values.reserve(indices.size());
  output_shapes.reserve(indices.size());
  for (auto index : indices) {
    XLATensorPtr tensor = tensors[index];
    ir_values.push_back(tensor->CurrentIrValue());
    output_shapes.push_back(MakeShapeWithDeviceLayout(
        tensor->shape(),
        static_cast<XlaDeviceType>(tensor->GetDevice().type())));
  }
  PostOrderData po_data = RunPostOrder(ir_values, &coll);
  torch::lazy::hash_t res_hash = torch::lazy::HashCombine(
      coll.hash, torch::lazy::Hash(po_data.parameter_sequence));
  DeviceContextArena::Get()->SaveOutputShapes(res_hash,
                                              std::move(output_shapes));
  DeviceContextArena::Get()->SaveGraphAsString(res_hash, tensors,
                                               &coll.indices);
  return res_hash;
}

void XLAGraphExecutor::MaybeDumpGraph(std::string name,
                                      torch::lazy::hash_t hash) {
  thread_local const std::string save_file = xla::sys_util::GetEnvOrdinalPath(
      "XLA_SAVE_TENSORS_FILE", "", GetCurrentDevice().ordinal());
  if (!save_file.empty()) {
    std::string graph = DeviceContextArena::Get()->GetGraphByHash(hash);
    if (graph.size() == 0) {
      return;
    }
    std::ofstream graph_file(save_file, std::ios_base::app);
    graph_file << "[" << name << "]\n" << graph << "\n";
  }
}

XLAGraphExecutor::ComputationCache* XLAGraphExecutor::GetComputationCache() {
  static const size_t kMaxCacheSize =
      xla::sys_util::GetEnvInt("XLA_COMPILATION_CACHE_SIZE", 1024);
  static ComputationCache* cache = new ComputationCache(kMaxCacheSize);
  return cache;
}

void XLAGraphExecutor::ClearPendingIrs(
    std::vector<XLATensorPtr> tensors,
    const torch::lazy::BackendDevice& device) {
  std::unordered_set<int64_t> tensor_ids;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensor_ids.insert(tensors[i]->GetUniqueId()).second &&
        tensors[i]->CurrentDataHandle() == nullptr) {
      torch::lazy::Value ir_value = tensors[i]->CurrentIrValue();
      if (ir_value) {
        xla::Shape shape = MakeShapeWithDeviceLayout(
            tensors[i]->shape(), static_cast<XlaDeviceType>(device.type()));
        torch::lazy::BackendDataPtr handle =
            WrapXlaData(xla::ComputationClient::Get()->CreateDataPlaceholder(
                device.toString(), std::move(shape)));
        tensors[i]->AssignIrValue(torch::lazy::Value());
        tensors[i]->data()->handle = handle;
        tensors[i]->data()->view = nullptr;
        tensors[i]->data()->tensor_data = c10::nullopt;
      }
    }
  }
}

XLAGraphExecutor::SyncTensorCollection XLAGraphExecutor::CollectSyncTensors(
    const std::vector<XLATensorPtr>& tensors, const SyncTensorsConfig& config) {
  tsl::profiler::TraceMe activity("CollectSyncTensors",
                                  tsl::profiler::TraceMeLevel::kInfo);
  xla::util::Unique<torch::lazy::BackendDevice> unique_device;
  for (size_t i = 0; i < tensors.size(); ++i) {
    unique_device.set(tensors[i]->GetDevice());
  }
  SyncTensorCollection coll;
  if (!unique_device) {
    return coll;
  }

  std::vector<at::Tensor> at_tensors;
  std::vector<std::string> devices;
  std::vector<XLATensor::ShardingSpecPtr> shardings;
  std::vector<size_t> at_tensor_index;
  std::unordered_set<int64_t> tensor_ids;
  // The force_ltc_data controls aliasing compilation, so effectively the same
  // graph with on/off force_ltc_data should not match, hash wise.
  coll.hash = torch::lazy::MHash(config.force_ltc_data);
  coll.config = config;
  coll.device = *unique_device;
  coll.indices.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensor_ids.insert(tensors[i]->GetUniqueId()).second &&
        // A tensor's xla_data might not be up to date if there is a view
        // associated with it. Make sure to sync those tensors here too.
        (tensors[i]->CurrentDataHandle() == nullptr ||
         (tensors[i]->data()->view != nullptr &&
          !tensors[i]->data()->view->IsUpToDate()))) {
      torch::lazy::Value ir_value = tensors[i]->CurrentIrValue();
      if (ir_value) {
        if (ShouldSyncIrValue(ir_value)) {
          // Add only tensors which need to be synced.
          coll.hash = torch::lazy::HashCombine(coll.hash, ir_value.hash());
          coll.indices.push_back(i);

          // `sharding_spec()` checks sharding equality. If IR node has no
          // sharding, then sync XLATensor sharding to the IR node. XLATensor's
          // sharding takes the precedence as the source of the truth.
          XLATensor::ShardingSpecPtr sharding = tensors[i]->sharding_spec();
          if (sharding) {
            dynamic_cast<XlaNode*>(ir_value.node.get())
                ->SetSharding(sharding->sharding);
          }
        }
      } else if (config.force_ltc_data) {
        // The tensor only has at::Tensor data. We need to queue it for a
        // device upload.
        c10::optional<at::Tensor> tensor_data = tensors[i]->CurrentTensorData();
        XLA_CHECK(tensor_data);
        at_tensors.push_back(*tensor_data);
        shardings.push_back(tensors[i]->sharding_spec());
        devices.push_back(tensors[i]->GetDevice().toString());
        at_tensor_index.push_back(i);
      }
    }
  }
  // Mix the hash with the resource domain hashes as compile handles are only
  // valid within a domain (usually a single host).
  coll.hash = torch::lazy::MHash(
      coll.hash,
      xla::ComputationClient::Get()->GetResourceDomain(coll.device.toString()));
  if (!at_tensors.empty()) {
    TORCH_LAZY_COUNTER("SyncTensorsToData", at_tensors.size());
    // Create data handles with shardings. If a tensor has a
    // sharding annotation, then a BackendDataPtr with PjRtShardedData is
    // returned; if there is no sharding annotation, then a BackendDataPtr with
    // PjRtData is returned.
    std::vector<torch::lazy::BackendDataPtr> handles =
        CreateTensorsData(at_tensors, shardings, devices);
    for (size_t i = 0; i < handles.size(); ++i) {
      // If we are here, it means that the IR torch::lazy::Value for the
      // tensor is not present. Also, we uploaded the at::Tensor data to the
      // device, but such data is still valid so we leave it live on the XLA
      // tensor (so that a following ToTensor() does not need to fetch it from
      // device).
      tensors[at_tensor_index[i]]->data()->handle = std::move(handles[i]);
    }
  }
  TF_VLOG(4) << "Tensors graph hash " << torch::lazy::HashToString(coll.hash)
             << " on device " << coll.device;
  return coll;
}

void XLAGraphExecutor::TensorCollectionBarrier(SyncTensorCollection* coll) {
  torch::lazy::LazyGraphExecutor::TensorCollectionBarrier(coll);
  // TODO(yeounoh) lock SPMD device
}

std::vector<torch::lazy::BackendDataPtr>
XLAGraphExecutor::ExecuteComputationWithBarrier(
    torch::lazy::hash_t hash, const std::vector<at::IValue>& graph_inputs,
    const torch::lazy::BackendDevice& device) {
  MaybeDumpGraph("dynamo", hash);
  auto cachedComputation =
      XLAGraphExecutor::Get()->GetComputationCache()->Get(hash);
  // TODO implement a fallback mechanism, or make sure those entries
  // never get kicked out
  XLA_CHECK(cachedComputation)
      << "Failed to get computation by hash " << torch::lazy::HashToString(hash)
      << ". Maybe the entry get "
         "kicked out of the LRU cache";

  // Create DataPlaceHolder that will get filled in async executions.
  std::vector<xla::Shape>* output_shapes =
      DeviceContextArena::Get()->GetOutputShapesByHash(hash);
  std::vector<torch::lazy::BackendDataPtr> placeholders;
  placeholders.reserve(output_shapes->size());
  for (const xla::Shape& shape : *output_shapes) {
    torch::lazy::BackendDataPtr handle =
        WrapXlaData(xla::ComputationClient::Get()->CreateDataPlaceholder(
            device.toString(), std::move(shape)));
    placeholders.push_back(handle);
  }

  SyncTensorCollection coll;
  coll.device = device;
  coll.unlocker = DeviceLockerArena::Get()->LockDevices({device});
  std::vector<torch::lazy::BackendDataPtr> arguments;
  {
    // GetXlaData must be called within a lock region, otherwise it might
    // extract the placeholder inserted by previous execution.
    TORCH_LAZY_TIMED("RunCachedGraphInputData");
    // setup the arguments
    int idx = 0;
    for (auto& ivalue : graph_inputs) {
      torch::lazy::BackendDataPtr dataptr;
      if (auto xla_tensor_ptr = bridge::TryGetXlaTensor(ivalue.toTensor())) {
        dataptr = xla_tensor_ptr->GetXlaData();
      } else {
        dataptr = torch_xla::TensorToXlaData(ivalue.toTensor(), device);
      }

      ++idx;
      arguments.push_back(dataptr);
    }
  }

  std::shared_ptr<XLAGraphExecutor::Async> async = std::make_shared<Async>(
      &coll, std::move(arguments), placeholders, std::move(cachedComputation));

  auto syncfn = [async, hash]() {
    TF_VLOG(3) << "Executing Dynamo IR graph hash "
               << torch::lazy::HashToString(hash) << " on device "
               << async->device << " ...";
    std::vector<torch::lazy::BackendDataPtr> results =
        torch::lazy::getBackend()->ExecuteComputation(
            async->cached_computation->computation, async->parameters_data,
            async->device);
    TF_VLOG(3) << "Executing Dynamo IR graph hash "
               << torch::lazy::HashToString(hash) << " on device "
               << async->device << " done!";
    // Updating placeholder with actual output handle.
    for (size_t i = 0; i < results.size(); ++i) {
      XLA_CHECK(async->tensors_data[i] != nullptr);
      async->tensors_data[i]->Assign(*results[i]);
    }
  };

  xla::env::ScheduleIoClosure(async->mwait.Completer(std::move(syncfn)));

  return placeholders;
}

std::vector<at::Tensor> XLAGraphExecutor::GetTensorsOpByOp(
    std::vector<XLATensorPtr>* tensors) {
  SyncTensorsConfig config;
  config.force_ltc_data = false;
  SyncTensorCollection coll = CollectSyncTensors(*tensors, config);
  std::vector<torch::lazy::BackendDataPtr> async_tensors_data;
  if (!coll.indices.empty()) {
    DebugUtil::SaveTensorsGraphInfo("GetTensorsOpByOp", *tensors,
                                    &coll.indices);

    std::vector<torch::lazy::Value> roots =
        CollectRoots(*tensors, coll.indices);
    TensorCollectionBarrier(&coll);
    async_tensors_data =
        OpByOpExecutor::Get()->Execute(roots, coll.device.toString(), {});
  }

  std::vector<torch::lazy::BackendDataPtr> tensors_data =
      GatherTensorsXlaData(*tensors, coll.indices, async_tensors_data);
  std::vector<xla::Literal> literals =
      xla::ComputationClient::Get()->TransferFromServer(
          UnwrapXlaData(tensors_data));

  return FetchTensors(tensors, literals, &coll.indices);
}

std::vector<at::Tensor> XLAGraphExecutor::GetTensorsFused(
    std::vector<XLATensorPtr>* tensors) {
  SyncTensorsConfig config;
  config.force_ltc_data = false;
  auto async = SyncTensorsGraphInternal(tensors, {}, config);
  if (async != nullptr) {
    async->mwait.Wait();
  }
  std::vector<torch::lazy::BackendDataPtr> tensors_data = GatherTensorsXlaData(
      *tensors, async != nullptr ? async->indices : absl::Span<const size_t>(),
      async != nullptr ? async->tensors_data
                       : absl::Span<const torch::lazy::BackendDataPtr>());

  // Execution is async in PJRT, so TransferFromServer may block until execution
  // completes. Release the GIL so other threads can proceed and unblock any
  // collective computations.
  // HACK: This method may be called outside of python (mainly in C++ tests) or
  // when the GIL is already released, so we must check both cases here. If
  // possible, prefer to release the GIL in the python bindings before copying
  // this pattern.
  PyThreadState* save = nullptr;
  // TODO(wcromar): Remove this setting when we are more confident
  static const bool release_gil =
      xla::sys_util::GetEnvBool("XLA_RELEASE_GIL_DURING_TRANSFER", true);
  if (release_gil && Py_IsInitialized() && PyGILState_Check()) {
    save = PyEval_SaveThread();
  }
  std::vector<xla::Literal> literals =
      xla::ComputationClient::Get()->TransferFromServer(
          UnwrapXlaData(tensors_data));
  if (save) {
    PyEval_RestoreThread(save);
  }

  return FetchTensors(tensors, literals,
                      async != nullptr ? &async->indices : nullptr);
}

XLAGraphExecutor::OpByOpAsync XLAGraphExecutor::SyncTensorsGraphOpByOp(
    std::vector<XLATensorPtr>* tensors, absl::Span<const std::string> devices,
    const SyncTensorsConfig& config) {
  struct Async {
    explicit Async(SyncTensorCollection coll,
                   std::vector<torch::lazy::BackendDataPtr> tensors_data,
                   std::vector<torch::lazy::Value> roots,
                   absl::Span<const std::string> devices)
        : coll(std::move(coll)),
          tensors_data(std::move(tensors_data)),
          roots(std::move(roots)),
          devices(devices.begin(), devices.end()) {}

    SyncTensorCollection coll;
    std::vector<torch::lazy::BackendDataPtr> tensors_data;
    std::vector<torch::lazy::Value> roots;
    std::vector<std::string> devices;
  };

  SyncTensorCollection coll = CollectSyncTensors(*tensors, config);
  DebugUtil::SaveTensorsGraphInfo("SyncTensorsGraphOpByOp", *tensors,
                                  &coll.indices);

  std::vector<torch::lazy::Value> roots = CollectRoots(*tensors, coll.indices);
  std::vector<torch::lazy::Value> ir_values;
  std::vector<torch::lazy::BackendDataPtr> tensor_data_vec;
  ExtractIRAndPrepareXlaData_(tensors, coll.config, coll.indices, ir_values,
                              tensor_data_vec);
  auto tensors_data =
      SetTensorData(tensors, coll.config, coll.indices, tensor_data_vec);
  TensorCollectionBarrier(&coll);
  auto async = std::make_shared<Async>(std::move(coll), std::move(tensors_data),
                                       std::move(roots), devices);
  auto syncfn = [async]() -> int {
    try {
      TF_VLOG(3) << "Executing (OpByOp) IR graph hash "
                 << torch::lazy::HashToString(async->coll.hash) << " on device "
                 << async->coll.device << " ...";
      std::vector<torch::lazy::BackendDataPtr> results =
          OpByOpExecutor::Get()->Execute(
              async->roots, async->coll.device.toString(), async->devices);
      TF_VLOG(3) << "Executing (OpByOp) IR graph hash "
                 << torch::lazy::HashToString(async->coll.hash) << " on device "
                 << async->coll.device << " done!";

      for (size_t i = 0; i < results.size(); ++i) {
        if (async->tensors_data[i] != nullptr) {
          async->tensors_data[i]->Assign(*results[i]);
        }
      }
    } catch (...) {
      for (auto& unlocker : async->coll.unlocker) {
        unlocker.SetStatus(std::current_exception());
      }
      throw;
    }
    return 0;
  };
  OpByOpAsync async_op(std::move(syncfn));
  return async_op.Schedule();
}

std::vector<torch::lazy::BackendDataPtr> XLAGraphExecutor::GatherTensorsXlaData(
    const std::vector<XLATensorPtr>& tensors, absl::Span<const size_t> indices,
    absl::Span<const torch::lazy::BackendDataPtr> tensors_data) {
  std::vector<torch::lazy::BackendDataPtr> result_tensors_data;
  std::unordered_map<int64_t, size_t> uid_index_map;
  size_t indices_index = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    int64_t tensor_id = tensors[i]->GetUniqueId();
    auto it = uid_index_map.find(tensor_id);
    if (it != uid_index_map.end()) {
      // Current tensor is a duplicate of a previously processed tensor that
      // had an IR XlaNode to sync. Get the XLA data from the tensor_data_map.
      result_tensors_data.push_back(result_tensors_data[it->second]);
    } else if (indices_index < indices.size() && i == indices[indices_index]) {
      // If we are at the current index (it means that the tensor at index
      // 'i' had an IR node to sync), use the XLA data held within the Async
      // object.
      uid_index_map.emplace(tensor_id, result_tensors_data.size());
      result_tensors_data.push_back(tensors_data[indices_index]);
      ++indices_index;
    } else if (!tensors[i]->CurrentTensorData()) {
      torch::lazy::BackendDataPtr handle = tensors[i]->CurrentDataHandle();
      XLA_CHECK(handle != nullptr);
      result_tensors_data.push_back(std::move(handle));
    }
  }
  return result_tensors_data;
}

std::vector<torch::lazy::Value> XLAGraphExecutor::CollectRoots(
    const std::vector<XLATensorPtr>& tensors,
    absl::Span<const size_t> indices) {
  std::vector<torch::lazy::Value> roots;
  roots.reserve(indices.size());
  for (auto index : indices) {
    roots.push_back(tensors.at(index)->CurrentIrValue());
  }
  return roots;
}

std::vector<XLATensor::ShardingSpecPtr> XLAGraphExecutor::CollectShardingSpecs(
    std::vector<XLATensorPtr>* tensors, absl::Span<const size_t> indices) {
  std::vector<XLATensor::ShardingSpecPtr> sharding_specs;
  sharding_specs.reserve(indices.size());
  for (const size_t index : indices) {
    XLATensorPtr& tensor = (*tensors)[index];
    sharding_specs.push_back(tensor->sharding_spec());
  }
  return sharding_specs;
}

std::vector<torch::lazy::BackendDataPtr> XLAGraphExecutor::SetTensorData(
    std::vector<XLATensorPtr>* tensors, const SyncTensorsConfig& config,
    absl::Span<const size_t> indices,
    const std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec) {
  tsl::profiler::TraceMe activity("SetTensorData",
                                  tsl::profiler::TraceMeLevel::kInfo);
  std::vector<torch::lazy::BackendDataPtr> tensors_data;
  tensors_data.reserve(indices.size());
  for (int i = 0; i < indices.size(); i++) {
    auto index = indices[i];
    XLATensorPtr& tensor = (*tensors)[index];
    // If the config.force_ltc_data flag is true, the purpose of this tensor
    // sync operation is to truncate the IR graph and materialize device data
    // in place of IR graph, on selected tensors. But since operation will
    // complete asynchronously, if a tensor does not already have device data,
    // we need to install a placeholder. Since at this point we hold a lock on
    // the device where the tensors reside (locks held within the coll
    // structure, and moved into the async variable), any other operation
    // trying to access the tensor's device data will have to wait until the
    // asynchronous operation completes.
    torch::lazy::BackendDataPtr handle = tensor->CurrentDataHandle();
    if (!handle && tensor->CurrentIrValue()) {
      handle = torch::lazy::getBackend()->GetComputationDataFromNode(
          tensor->CurrentIrValue().node.get());
    }

    if (handle == nullptr && config.force_ltc_data) {
      handle = tensor_data_vec[i];
      // Note: We are not using SetXlaData method here since that method
      // resets the ir_value. We have already done the resetting as part
      // of ExtractIRAndPrepareXlaData_ to overlap with previous execution.
      tensor->data()->handle = handle;
      tensor->data()->view = nullptr;
      tensor->data()->tensor_data = c10::nullopt;
    }
    tensors_data.emplace_back(std::move(handle));
  }
  return tensors_data;
}

void XLAGraphExecutor::ExtractIRAndPrepareXlaData_(
    std::vector<XLATensorPtr>* tensors, const SyncTensorsConfig& config,
    const absl::Span<const size_t> indices,
    std::vector<torch::lazy::Value>& ir_values,
    std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec) {
  tsl::profiler::TraceMe activity("ExtractIRAndPrepareXlaData_",
                                  tsl::profiler::TraceMeLevel::kInfo);
  ir_values.reserve(indices.size());
  tensor_data_vec.reserve(indices.size());
  for (auto index : indices) {
    XLATensorPtr& tensor = (*tensors)[index];
    torch::lazy::Value ir_value = tensor->CurrentIrValue();

    ir_values.push_back(ir_value);
    const torch::lazy::BackendDevice& tensor_device = tensor->GetDevice();
    xla::Shape shape = MakeShapeWithDeviceLayout(
        tensor->shape(), static_cast<XlaDeviceType>(tensor_device.type()));
    torch::lazy::BackendDataPtr handle =
        WrapXlaData(xla::ComputationClient::Get()->CreateDataPlaceholder(
            tensor_device.toString(), std::move(shape)));
    // Create sharded data placeholder, this will be used to
    // hold the corresponding computation results.
    if (tensor->sharding_spec()) {
      auto sharding = tensor->sharding_spec();
      if (!sharding->shape.has_value()) {
        sharding->shape = tensor->shape();
      }
      handle = WrapXlaData(xla::ComputationClient::Get()->WrapDataShards(
          {UnwrapXlaData(handle)}, GetVirtualDevice().toString(),
          sharding->shape.value(), sharding->sharding));
    }
    tensor_data_vec.push_back(handle);
    if (tensor->CurrentDataHandle() == nullptr && config.force_ltc_data) {
      tensor->AssignIrValue(torch::lazy::Value());
    }
  }
}

std::vector<at::Tensor> XLAGraphExecutor::FetchTensors(
    std::vector<XLATensorPtr>* tensors, absl::Span<const xla::Literal> literals,
    const std::vector<size_t>* indices) {
  std::vector<at::Tensor> results;
  size_t literals_index = 0;
  size_t sync_index = 0;
  results.reserve(tensors->size());
  for (size_t i = 0; i < tensors->size(); ++i) {
    if (indices != nullptr && sync_index < indices->size() &&
        i == (*indices)[sync_index]) {
      results.push_back(MakeTensorFromXlaLiteral(literals[literals_index],
                                                 (*tensors)[i]->dtype()));
      ++literals_index;
      ++sync_index;
    } else {
      c10::optional<at::Tensor> tensor_data =
          (*tensors)[i]->CurrentTensorData();
      if (tensor_data) {
        results.push_back(*tensor_data);
      } else {
        XLA_CHECK_LT(literals_index, literals.size());
        results.push_back(MakeTensorFromXlaLiteral(literals[literals_index],
                                                   (*tensors)[i]->dtype()));
        ++literals_index;
      }
    }
  }
  return results;
}

std::shared_ptr<XLAGraphExecutor::Async>
XLAGraphExecutor::ScheduleSyncTensorsGraph(
    SyncTensorCollection* coll,
    std::vector<torch::lazy::BackendDataPtr> parameters_data,
    std::vector<torch::lazy::BackendDataPtr> tensors_data,
    std::vector<XLATensor::ShardingSpecPtr> sharding_specs,
    ComputationCache::TypePtr cached_computation) {
  tsl::profiler::TraceMe activity("ScheduleSyncTensorsGraph",
                                  tsl::profiler::TraceMeLevel::kInfo);
  TensorCollectionBarrier(coll);
  std::shared_ptr<XLAGraphExecutor::Async> async = std::make_shared<Async>(
      coll, std::move(parameters_data), std::move(tensors_data),
      std::move(cached_computation));
  auto syncfn = [async, hash = coll->hash, sharding_specs = sharding_specs]() {
    try {
      std::vector<torch::lazy::BackendDataPtr> results;
      // Execute replicated if the compiled computation is partitioned.
      if (async->cached_computation->is_sharded) {
        std::vector<std::string> devices =
            xla::ComputationClient::Get()->GetLocalDevices();
        std::vector<std::vector<xla::ComputationClient::DataPtr>>
            device_arguments = ShardingUtil::InputHandler(
                UnwrapXlaData(async->parameters_data), devices);
        xla::ComputationClient::ExecuteReplicatedOptions execute_options;
        TF_VLOG(3) << "Executing IR graph hash "
                   << torch::lazy::HashToString(hash)
                   << " on devices: " << absl::StrJoin(devices, ",");
        // OutputHandler creates sharded data for sharded
        // tensor results. Both sharded and unsharded results should be
        // "Assign"ed to the corresponding data placeholders.
        std::vector<xla::ComputationClient::DataPtr> outputs =
            ShardingUtil::OutputHandler(
                xla::ComputationClient::Get()->ExecuteReplicated(
                    *async->cached_computation->computation
                         ->client_computation(),
                    device_arguments, devices, execute_options),
                sharding_specs);
        results = WrapXlaData(outputs);
        TF_VLOG(3) << "Executing IR graph hash "
                   << torch::lazy::HashToString(hash)
                   << " on devices: " << absl::StrJoin(devices, ",")
                   << " done!";
      } else {
        TF_VLOG(3) << "Executing IR graph hash "
                   << torch::lazy::HashToString(hash) << " on device "
                   << async->device << " ...";
        results = torch::lazy::getBackend()->ExecuteComputation(
            async->cached_computation->computation, async->parameters_data,
            async->device);
        TF_VLOG(3) << "Executing IR graph hash "
                   << torch::lazy::HashToString(hash) << " on device "
                   << async->device << " done!";
      }
      for (size_t i = 0; i < results.size(); ++i) {
        if (async->tensors_data[i] != nullptr) {
          async->tensors_data[i]->Assign(*results[i]);
        } else {
          async->tensors_data[i] = std::move(results[i]);
        }
      }
    } catch (...) {
      // There are two paths of discovery of an exception happening on an
      // asynchronous task. One happens if the creator of the asynchronous task
      // explicitly waits for completion, in which case the exception will be
      // thrown from the Wait() API. Re-throwing the exception below makes sure
      // this will be captured by the completer function created below, and
      // surfaced by the Wait() API. But we also need to surface the exception
      // even in case the caller does not wait, and that is accomplished by
      // setting the unlockers status. In that case the exception will be
      // surfaced when the user tries to acquire the device locks the next time.
      for (auto& unlocker : async->unlocker) {
        unlocker.SetStatus(std::current_exception());
      }
      throw;
    }
  };

  xla::env::ScheduleIoClosure(async->mwait.Completer(std::move(syncfn)));
  return async;
}

std::shared_ptr<XLAGraphExecutor::Async>
XLAGraphExecutor::ScheduleSyncTensorsGraph(
    std::vector<XLATensorPtr>* tensors, SyncTensorCollection* coll,
    std::vector<torch::lazy::BackendDataPtr> parameters_data,
    std::string device, ComputationCache::TypePtr cached_computation,
    const std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec) {
  auto tensors_data =
      SetTensorData(tensors, coll->config, coll->indices, tensor_data_vec);
  auto sharding_specs = CollectShardingSpecs(tensors, coll->indices);
  return ScheduleSyncTensorsGraph(
      coll, std::move(parameters_data), std::move(tensors_data),
      std::move(sharding_specs), std::move(cached_computation));
}

XLAGraphExecutor::PostOrderData XLAGraphExecutor::RunPostOrder(
    const std::vector<torch::lazy::Value>& ir_values,
    SyncTensorCollection* coll) {
  tsl::profiler::TraceMe activity("RunPostOrder",
                                  tsl::profiler::TraceMeLevel::kInfo);
  return torch::lazy::LazyGraphExecutor::RunPostOrder(ir_values, coll);
}

XLAGraphExecutor::ComputationCache::TypePtr
XLAGraphExecutor::LookupCachedCompile(const torch::lazy::hash_t& hash) {
  ComputationCache::TypePtr cached_computation =
      GetComputationCache()->Get(hash);
  if (cached_computation == nullptr) {
    TORCH_LAZY_COUNTER("UncachedCompile", 1);
    return nullptr;
  }
  TF_VLOG(5) << "Graph hash " << torch::lazy::HashToString(hash)
             << " is computation hash "
             << torch::lazy::HashToString(torch::lazy::Hash(
                    cached_computation->computation->computation()
                        .proto()
                        .SerializeAsString()));
  TORCH_LAZY_COUNTER("CachedCompile", 1);
  return cached_computation;
}

std::shared_ptr<XLAGraphExecutor::Async> XLAGraphExecutor::TryRunCachedSync(
    std::vector<XLATensorPtr>* tensors, SyncTensorCollection* coll,
    PostOrderData* po_data,
    const std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec) {
  ComputationCache::TypePtr cached_computation =
      LookupCachedCompile(coll->hash);
  if (cached_computation == nullptr) {
    return nullptr;
  }
  TORCH_LAZY_VALUE_METRIC("TensorsGraphSize", po_data->post_order.size());
  TF_VLOG(5) << "TensorsGraphSize=" << po_data->post_order.size();

  return ScheduleSyncTensorsGraph(
      tensors, coll, std::move(po_data->parameters_data),
      coll->device.toString(), std::move(cached_computation), tensor_data_vec);
}

std::vector<std::pair<int64_t, int64_t>>
XLAGraphExecutor::BuildInputOutputAliases(
    const std::vector<XLATensorPtr>& tensors, absl::Span<const size_t> indices,
    LoweringContext* lowering_ctx) {
  std::unordered_map<int64_t, size_t> output_tensor_id_map;
  std::vector<std::pair<int64_t, int64_t>> input_output_alias_pair;
  // tensors[indices] represent all tensors that needs to be updated after
  // the execution. We can only alias the current buffer of these tensors since
  // those buffers are no longer needed after execution.
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t tensor_index = indices[i];
    int64_t tensor_id = tensors[tensor_index]->data()->alias_id;
    output_tensor_id_map[tensor_id] = i;
  }
  const auto& parameters_data = lowering_ctx->GetParametersData();
  std::vector<ssize_t> alias_map(indices.size(), -1);
  for (size_t i = 0; i < parameters_data.size(); ++i) {
    auto* data_info =
        static_cast<torch::lazy::LazyGraphExecutor::DeviceDataInfo*>(
            parameters_data[i]->info());
    if (data_info != nullptr && !data_info->read_only) {
      auto it = output_tensor_id_map.find(data_info->tensor_id);
      // Parameter buffer's TensorId in output_tensor_id_map means
      // this buffer is not needed after execution since XLATensor will get a
      // new buffer.
      if (it != output_tensor_id_map.end()) {
        size_t output_index = it->second;
        xla::XlaOp root = lowering_ctx->GetResult(output_index);
        const xla::Shape& root_shape = XlaHelpers::ShapeOfXlaOp(root);
        auto parameter_data_shape = UnwrapXlaData(parameters_data[i])->shape();
        // Need to check whether existing buffer and the new value has the same
        // shape and the existing buffer has not been aliased before aliasing
        // the existing and new buffer.
        if (parameter_data_shape == root_shape && alias_map[output_index] < 0) {
          // parameter is not a tuple so param_index will always be {}
          lowering_ctx->builder()->SetUpAlias(
              {/*output_index=*/static_cast<int64_t>(output_index)},
              /*param_number=*/i, /*param_index=*/{});
          alias_map[output_index] = i;
          input_output_alias_pair.push_back(std::make_pair(i, output_index));

          TF_VLOG(6) << "Aliased paramter " << i << " with output "
                     << output_index << ": " << parameter_data_shape;
        }
      }
    }
  }
  TORCH_LAZY_VALUE_METRIC("InputOutputAliasCount",
                          input_output_alias_pair.size());
  return input_output_alias_pair;
}

XLAGraphExecutor::CompilationResult XLAGraphExecutor::Compile(
    const std::vector<XLATensorPtr>& tensors,
    absl::Span<const std::string> devices, const SyncTensorCollection& coll,
    PostOrderData* po_data, const std::vector<torch::lazy::Value>& ir_values) {
  tsl::profiler::TraceMe activity(
      [&] {
        return tsl::profiler::TraceMeEncode(
            "XLAGraphExecutor::Compile",
            {{"graph_hash", torch::lazy::HashToString(coll.hash)}});
      },
      tsl::profiler::TraceMeLevel::kInfo);
  static const bool enable_aliasing =
      xla::sys_util::GetEnvBool("XLA_ENABLE_PARAM_ALIASING", true);
  static const size_t parameter_wrapping_threadshold =
      xla::sys_util::GetEnvInt("XLA_PARAMETER_WRAPPING_THREADSHOLD", 3200);
  static const bool using_pjrt =
      xla::sys_util::GetEnvString("PJRT_DEVICE", "").size() > 0;
  LoweringContext lowering_ctx("SyncTensorsGraph", coll.device,
                               po_data->post_order,
                               std::move(po_data->emission_map));
  for (auto ir_value : ir_values) {
    xla::XlaOp root = lowering_ctx.GetOutputOp(
        torch::lazy::Output(ir_value.node.get(), ir_value.index));
    lowering_ctx.AddResult(root);
  }
  // Annotate HLO sharding selectively in the compuation.
  bool is_sharded = ShardingUtil::SetHloSharding(&lowering_ctx);

  std::vector<std::pair<int64_t, int64_t>> input_output_alias_pair;
  // TODO(yeounoh) aliasing is disabled for partitioned computation,
  // since the current aliasing compares the unpartitioned input and output
  // shapes which can lead to an incorrect aliasing pairs if sharded.
  if (enable_aliasing && coll.config.sync_ltc_data &&
      coll.config.force_ltc_data && !is_sharded) {
    // We can only alias at the step barrier, when force_ltc_data is true.
    // Consider the case:
    //   1. Tensor A(DEVICE_DATA)
    //   2. Tensor B = A + 0.9
    //   3. A += 0.4
    // If we activate aliasing for A's graph, and we do:
    //   print(A)
    //   print(A)
    // The first print will update DEVICE_DATA' with DEVICE_DATA+0.4, and the
    // second print will again update DEVICE_DATA" with DEVICE_DATA'+0.4, which
    // will lead to incorrect results.
    // We cannot normally turn A's state into DEVICE_DATA, as if any of the
    // sources is a view, this will not lead to correct results (as A's value
    // taken at different times need to reflect view source changes):
    //   1. Tensor A = some_graph_with_view_source(V)
    //   2. print(A)
    //   3. V += 1
    //   4. print(A)
    // The second print should reflect the new value due to V's changes.
    // Also in the first example, unless we are doing a step barrier and hence
    // include all live tensors, if the B value is not part of the graph, it
    // will later fetch the new value of A, which is incorrect.
    // But, when we issue a step barrier (force_ltc_data == true) we have to
    // turn everything into DEVICE_DATA, so we can activate aliasing.
    input_output_alias_pair =
        BuildInputOutputAliases(tensors, coll.indices, &lowering_ctx);
  }

  xla::XlaComputation computation = ConsumeValue(lowering_ctx.BuildXla());
  xla::ProgramShape program_shape = ConsumeValue(computation.GetProgramShape());

  bool should_wrap_parameter =
      (program_shape.parameters_size() >= parameter_wrapping_threadshold) &&
      using_pjrt;
  if (should_wrap_parameter) {
    TF_VLOG(3) << "Wrapping graph with " << program_shape.parameters_size()
               << " parameters. Threadshold = "
               << parameter_wrapping_threadshold;
    computation = ConsumeValue(XlaHelpers::WrapXlaComputation(
        computation, program_shape.parameters(), input_output_alias_pair));
    program_shape = ConsumeValue(computation.GetProgramShape());
  }
  xla::Shape shape = MakeShapeWithDeviceLayout(
      program_shape.result(), static_cast<XlaDeviceType>(coll.device.type()));

  std::vector<xla::ComputationClient::CompileInstance> instances;
  instances.push_back({std::move(computation), coll.device.toString(),
                       xla::ComputationClient::Get()->GetCompilationDevices(
                           coll.device.toString(), devices),
                       &shape, should_wrap_parameter, is_sharded});

  TF_VLOG(3) << "Compiling IR graph hash "
             << torch::lazy::HashToString(coll.hash) << " on device "
             << coll.device << " ...";
  std::vector<std::shared_ptr<xla::ComputationClient::Computation>>
      computations =
          xla::ComputationClient::Get()->Compile(std::move(instances));
  TF_VLOG(3) << "Compiling IR graph hash "
             << torch::lazy::HashToString(coll.hash) << " on device "
             << coll.device << " done!";
  TF_VLOG(5) << "Compiled program shape "
             << computations.front()->program_shape().ToString() << std::endl;
  TF_VLOG(5)
      << "Graph hash " << torch::lazy::HashToString(coll.hash)
      << " is computation hash "
      << torch::lazy::HashToString(torch::lazy::Hash(
             computations.front()->computation().proto().SerializeAsString()));
  if (should_wrap_parameter) {
    XLA_CHECK_EQ(program_shape.parameters_size(), 1);
    XLA_CHECK_EQ(program_shape.parameters()[0].tuple_shapes_size(),
                 po_data->parameters_data.size());
  } else {
    XLA_CHECK_EQ(program_shape.parameters_size(),
                 po_data->parameters_data.size());
  }

  return {/*device=*/coll.device,
          /*emitted_nodes=*/lowering_ctx.GetEmittedNodeCount(),
          /*computation=*/
          std::make_shared<Computation>(std::move(computations.front())),
          /*parameters_data=*/std::move(po_data->parameters_data),
          /*is_sharded=*/is_sharded};
}

std::shared_ptr<XLAGraphExecutor::Async>
XLAGraphExecutor::SyncTensorsGraphInternal(
    std::vector<XLATensorPtr>* tensors, absl::Span<const std::string> devices,
    const SyncTensorsConfig& config, bool warm_up_cache_only) {
  tsl::profiler::TraceMe activity("SyncTensorsGraphInternal",
                                  tsl::profiler::TraceMeLevel::kInfo);
  SyncTensorCollection coll = CollectSyncTensors(*tensors, config);
  if (coll.indices.empty()) {
    /* Enure previous execution is complete before exiting this
     * function */
    TensorCollectionBarrier(&coll);
    return nullptr;
  }
  DebugUtil::SaveTensorsGraphInfo("ScheduleSyncTensorsGraph", *tensors,
                                  &coll.indices);
  std::vector<torch::lazy::Value> ir_values;
  std::vector<torch::lazy::BackendDataPtr> tensor_data_vec;
  ExtractIRAndPrepareXlaData_(tensors, coll.config, coll.indices, ir_values,
                              tensor_data_vec);
  PostOrderData po_data = RunPostOrder(ir_values, &coll);
  coll.hash = torch::lazy::HashCombine(
      coll.hash, torch::lazy::Hash(po_data.parameter_sequence));
  TF_VLOG(4) << "Parameter sequence graph hash "
             << torch::lazy::HashToString(coll.hash);
  std::shared_ptr<Async> async =
      TryRunCachedSync(tensors, &coll, &po_data, tensor_data_vec);
  if (async != nullptr) {
    return async;
  }
  CompilationResult compile_result =
      Compile(*tensors, devices, coll, &po_data, ir_values);
  TORCH_LAZY_VALUE_METRIC("TensorsGraphSize", compile_result.emitted_nodes);
  TF_VLOG(5) << "TensorsGraphSize=" << compile_result.emitted_nodes;

  auto cached_computation = std::make_shared<CachedComputation>(
      std::move(compile_result.computation), compile_result.is_sharded);
  GetComputationCache()->Add(coll.hash, cached_computation);

  if (warm_up_cache_only) {
    return nullptr;
  } else {
    return ScheduleSyncTensorsGraph(
        tensors, &coll, std::move(compile_result.parameters_data),
        compile_result.device.toString(), std::move(cached_computation),
        tensor_data_vec);
  }
}

}  // namespace torch_xla
