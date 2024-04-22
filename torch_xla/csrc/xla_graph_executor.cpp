#include "torch_xla/csrc/xla_graph_executor.h"

#include <Python.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/unique.h>
#include <torch/csrc/lazy/core/util.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <functional>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/dtype.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/layout_manager.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/cast.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/ops/dynamic_ir.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/expand_symint.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/ops/view.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/runtime/cache.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/runtime.h"
#include "torch_xla/csrc/runtime/stablehlo_helper.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/runtime/thread_pool.h"
#include "torch_xla/csrc/runtime/xla_util.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/xla_backend_impl.h"
#include "torch_xla/csrc/xla_sharding_util.h"
#include "tsl/platform/errors.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"

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
      runtime::sys_util::GetEnvOrdinalPath(
          "XLA_SAVE_TENSORS_FILE", "", bridge::GetCurrentDevice().ordinal()) !=
      "";
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
      GetDeviceData(value, MaybeUpcastToHostTorchType(type), device);
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

torch::lazy::Value XLAGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value, const xla::Shape& shape,
    SymIntElements size_elements,
    c10::optional<at::ScalarType> logical_element_type,
    const torch::lazy::BackendDevice& device) {
  xla::PrimitiveType primitive_type =
      logical_element_type
          ? MakeXlaPrimitiveType(*logical_element_type, &device)
          : shape.element_type();
  torch::lazy::Value ir_value =
      GetIrValueForScalar(value, primitive_type, device);
  return torch::lazy::MakeNode<ExpandSymInt>(ir_value, size_elements);
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
    const std::vector<XLATensorPtr>& tensors, EmitMode mode) {
  std::vector<torch::lazy::Value> ir_values;
  for (auto& tensor : tensors) {
    torch::lazy::Value ir_value = tensor->CurrentIrValue();
    if (ir_value) {
      ir_values.push_back(std::move(ir_value));
    }
  }
  return !ir_values.empty()
             ? DumpUtil::ToHlo(ir_values, bridge::GetCurrentDevice(), mode)
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
  SyncTensorsConfig config;
  config.sync_ltc_data = sync_ltc_data;
  if (warm_up_cache_only) {
    config.force_ltc_data = false;
  }
  auto async =
      SyncTensorsGraphInternal(tensors, devices, config, warm_up_cache_only);
  if (wait && async != nullptr && !warm_up_cache_only) {
    async->mwait.Wait();
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
  // remain as XLA_COUNTER to support
  // runtime::metrics::CreatePerformanceReport(). For more information, see
  // NOTE: [TORCH_LAZY_COUNTER v.s. XLA_COUNTER].
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
    if (UseVirtualDevice()) {
      wait_devices.insert(ParseDeviceString("SPMD:0"));
    } else {
      for (auto& device_str :
           runtime::GetComputationClient()->GetLocalDevices()) {
        wait_devices.insert(ParseDeviceString(device_str));
      }
    }
  }
  // The DeviceLockerArena::Get()->LockDevices() API returns a vector of
  // torch::lazy::ExceptionCleanup object, which is going to be freed
  // immediately, turning this operation into a lock barrier.
  DeviceLockerArena::Get()->LockDevices(wait_devices);
  TF_VLOG(4) << "XLAGraphExecutor::WaitDeviceOps completed";
}

std::vector<at::Tensor> XLAGraphExecutor::GetTensors(
    std::vector<XLATensorPtr>* tensors) {
  TF_VLOG(4) << "Trying to get the value of " << tensors->size()
             << " tensor(s)";
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

  std::vector<xla::Literal> literals = ReleaseGilAndTransferData(tensors_data);

  return FetchTensors(tensors, literals,
                      async != nullptr ? &async->indices : nullptr);
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
  thread_local const std::string save_file =
      runtime::sys_util::GetEnvOrdinalPath(
          "XLA_SAVE_TENSORS_FILE", "", bridge::GetCurrentDevice().ordinal());
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
      runtime::sys_util::GetEnvInt("XLA_COMPILATION_CACHE_SIZE", 1024);
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
      // Only clear the IR that is not a DeviceData Node.
      if (ir_value) {
        DeviceData* device_data =
            torch_xla::DeviceData::Cast(ir_value.node.get());
        if (device_data != nullptr) {
          tensors[i]->data()->handle = device_data->data();
        } else {
          xla::Shape shape = MakeShapeWithDeviceLayout(
              tensors[i]->shape(), static_cast<XlaDeviceType>(device.type()));
          torch::lazy::BackendDataPtr handle =
              runtime::GetComputationClient()->CreateDataPlaceholder(
                  device.toString(), std::move(shape));
          tensors[i]->data()->handle = handle;
          TF_VLOG(4) << "Replacing the IR " << ir_value.node.get()->ToString()
                     << " of Tensor with ID " << tensors[i]->GetUniqueId()
                     << " with placeholder";
        }
        tensors[i]->AssignIrValue(torch::lazy::Value());
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
  torch::lazy::Unique<torch::lazy::BackendDevice> unique_device;
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
                ->SetSharding(sharding->sharding, ir_value.index);
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
  tsl::profiler::TraceMe activity("TensorCollectionBarrier",
                                  tsl::profiler::TraceMeLevel::kInfo);
  TF_VLOG(4) << "waiting barrier for device " << coll->device.toString()
             << " start";
  torch::lazy::LazyGraphExecutor::TensorCollectionBarrier(coll);
  // TODO(yeounoh) lock SPMD device
  TF_VLOG(4) << "waiting barrier for device " << coll->device.toString()
             << " done";
}

std::vector<torch::lazy::BackendDataPtr>
XLAGraphExecutor::ExecuteComputationWithBarrier(
    torch::lazy::hash_t hash, const std::vector<at::IValue>& graph_inputs,
    const torch::lazy::BackendDevice& device) {
  tsl::profiler::TraceMe activity("ExecuteComputationWithBarrier",
                                  tsl::profiler::TraceMeLevel::kInfo);
  MaybeDumpGraph("dynamo", hash);
  if (runtime::sys_util::GetEnvBool("PT_XLA_DEBUG", false)) {
    DebugUtil::analyze_graph_execution_python_frame(
        /*from_dynamo_executation=*/true);
  }
  auto cachedComputation =
      XLAGraphExecutor::Get()->GetComputationCache()->Get(hash);
  TF_VLOG(5) << "Cached computation (hash: " << torch::lazy::HashToString(hash)
             << ") is_sharded=" << cachedComputation->is_sharded << std::endl;

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

  std::vector<XLATensor::ShardingSpecPtr> sharding_specs;
  if (static_cast<XlaDeviceType>(device.type()) == XlaDeviceType::SPMD) {
    sharding_specs =
        std::vector<XLATensor::ShardingSpecPtr>(output_shapes->size());
    // TODO(JackCaoG): Use LRU cache and add same cache to non-dynamo path.
    static std::unordered_map<torch::lazy::hash_t,
                              std::vector<XLATensor::ShardingSpecPtr>,
                              torch::lazy::HashReducer>
        output_sharding_hash;
    // For any given graph(each hash correspodning to one graph) there is only
    // one output sharding. We can cache this sharding here to avoid retrive
    // the sharding from the computation every time.
    if (output_sharding_hash.find(hash) == output_sharding_hash.end()) {
      TORCH_LAZY_COUNTER("UncachedOutputSharding", 1);
      output_sharding_hash[hash] = ShardingUtil::GetOutputSharding(
          output_shapes, cachedComputation->computation, device);
    }
    placeholders =
        ShardingUtil::CreateShardedPlaceholder(output_sharding_hash[hash]);
  } else {
    for (const xla::Shape& shape : *output_shapes) {
      torch::lazy::BackendDataPtr handle =
          runtime::GetComputationClient()->CreateDataPlaceholder(
              device.toString(), std::move(shape));
      placeholders.push_back(handle);
    }
  }

  SyncTensorCollection coll;
  coll.device = device;
  {
    tsl::profiler::TraceMe activity("DeviceBarrier",
                                    tsl::profiler::TraceMeLevel::kInfo);
    TF_VLOG(5) << "Lock device " << device.toString() << "...";
    coll.unlocker = DeviceLockerArena::Get()->LockDevices({device});
    TF_VLOG(5) << "Locking device " << device.toString() << " Done!";
  }

  std::vector<torch::lazy::BackendDataPtr> arguments;
  {
    // GetXlaData must be called within a lock region, otherwise it might
    // extract the placeholder inserted by previous execution.
    TORCH_LAZY_TIMED("RunCachedGraphInputData");
    // setup the arguments
    for (auto& ivalue : graph_inputs) {
      torch::lazy::BackendDataPtr dataptr;
      if (auto xla_tensor_ptr = bridge::TryGetXlaTensor(ivalue.toTensor())) {
        dataptr = xla_tensor_ptr->GetXlaData();
      } else {
        XLA_CHECK(device.type() != (int8_t)XlaDeviceType::SPMD)
            << "SPMD device data should already be on the XLA backend "
               "(XLATensor).";
        dataptr = torch_xla::TensorToXlaData(ivalue.toTensor(), device);
      }
      arguments.push_back(dataptr);
    }
  }

  std::shared_ptr<XLAGraphExecutor::Async> async = std::make_shared<Async>(
      &coll, std::move(arguments), placeholders, std::move(cachedComputation));

  auto syncfn = [async, hash, sharding_specs]() {
    try {
      tsl::profiler::TraceMe activity("ExecuteComputationWithBarrier_syncfn",
                                      tsl::profiler::TraceMeLevel::kInfo);
      TF_VLOG(3) << "Executing Dynamo IR graph hash "
                 << torch::lazy::HashToString(hash) << " on device "
                 << async->device << " ...";

      std::vector<torch::lazy::BackendDataPtr> results;
      if (async->cached_computation->is_sharded) {
        std::vector<std::string> devices =
            runtime::GetComputationClient()->GetLocalDevices();
        std::vector<std::vector<runtime::ComputationClient::DataPtr>>
            device_arguments = ShardingUtil::InputHandler(
                UnwrapXlaData(async->parameters_data), devices);
        runtime::ComputationClient::ExecuteReplicatedOptions execute_options;
        // OutputHandler creates sharded data for sharded
        // tensor results. Both sharded and unsharded results should be
        // "Assign"ed to the corresponding data placeholders.
        std::vector<runtime::ComputationClient::DataPtr> outputs =
            ShardingUtil::OutputHandler(
                runtime::GetComputationClient()->ExecuteReplicated(
                    *async->cached_computation->computation, device_arguments,
                    devices, execute_options),
                sharding_specs);
        results = WrapXlaData(outputs);
        TF_VLOG(3) << "Executing Dynamo IR sharded graph hash "
                   << torch::lazy::HashToString(hash) << " on devices "
                   << absl::StrJoin(devices, ",") << " done!";
      } else {
        results = torch::lazy::getBackend()->ExecuteComputation(
            async->cached_computation->computation, async->parameters_data,
            async->device);
        TF_VLOG(3) << "Executing Dynamo IR graph hash "
                   << torch::lazy::HashToString(hash) << " on device "
                   << async->device << " done!";
      }

      // Updating placeholder with actual output handle.
      {
        tsl::profiler::TraceMe activity("update_placeholder",
                                        tsl::profiler::TraceMeLevel::kInfo);
        for (size_t i = 0; i < results.size(); ++i) {
          XLA_CHECK(async->tensors_data[i] != nullptr);
          async->tensors_data[i]->Assign(*results[i]);
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

  runtime::env::ScheduleIoClosure(async->mwait.Completer(std::move(syncfn)));

  return placeholders;
}

std::vector<torch::lazy::BackendDataPtr> XLAGraphExecutor::ExecuteStablehlo(
    std::string bytecode, const std::vector<at::IValue>& graph_inputs,
    const torch::lazy::BackendDevice& device) {
  // Convert StableHLO to HLO for XLA compilation.
  // TODO(lsy323): Pass StableHLO to PjrtComputationClient for compilation
  // after StableHLO compilation API is added in ComputationClient.
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::stablehlo::deserializePortableArtifact(bytecode, &context);
  mlir::ModuleOp mlir_module = *module;
  xla::HloProto hlo_proto;
  runtime::ConvertStableHloToHlo(&mlir_module, &context, &hlo_proto);
  xla::HloModuleProto* hlo_module_proto = hlo_proto.mutable_hlo_module();
  xla::XlaComputation computation(*hlo_module_proto);

  // Get program output shape.
  // TODO(lsy323): Get shape info from MLIR Module.
  xla::ProgramShape program_shape = ConsumeValue(computation.GetProgramShape());
  xla::Shape shape = MakeShapeWithDeviceLayout(
      program_shape.result(), static_cast<XlaDeviceType>(device.type()));

  std::vector<runtime::ComputationClient::CompileInstance> instances;
  instances.emplace_back(
      std::move(computation), device.toString(),
      runtime::GetComputationClient()->GetCompilationDevices(
          device.toString(),
          runtime::GetComputationClient()->GetLocalDevices()),
      &shape);
  std::vector<std::shared_ptr<runtime::ComputationClient::Computation>>
      computations =
          runtime::GetComputationClient()->Compile(std::move(instances));

  std::vector<torch::lazy::BackendDataPtr> arguments;
  {
    // GetXlaData must be called within a lock region, otherwise it might
    // extract the placeholder inserted by previous execution.
    // setup the arguments
    for (auto& ivalue : graph_inputs) {
      torch::lazy::BackendDataPtr dataptr;
      if (auto xla_tensor_ptr = bridge::TryGetXlaTensor(ivalue.toTensor())) {
        dataptr = xla_tensor_ptr->GetXlaData();
      } else {
        dataptr = torch_xla::TensorToXlaData(ivalue.toTensor(), device);
      }
      arguments.push_back(dataptr);
    }
  }

  std::vector<runtime::ComputationClient::DataPtr> result_data =
      runtime::GetComputationClient()->ExecuteComputation(
          *computations[0], UnwrapXlaData(arguments), device.toString());

  return WrapXlaData(result_data);
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
        runtime::GetComputationClient()->CreateDataPlaceholder(
            tensor_device.toString(), std::move(shape));
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
            runtime::GetComputationClient()->GetLocalDevices();
        std::vector<std::vector<runtime::ComputationClient::DataPtr>>
            device_arguments = ShardingUtil::InputHandler(
                UnwrapXlaData(async->parameters_data), devices);
        runtime::ComputationClient::ExecuteReplicatedOptions execute_options;
        TF_VLOG(3) << "Executing IR graph hash "
                   << torch::lazy::HashToString(hash)
                   << " on devices: " << absl::StrJoin(devices, ",");
        // OutputHandler creates sharded data for sharded
        // tensor results. Both sharded and unsharded results should be
        // "Assign"ed to the corresponding data placeholders.
        std::vector<runtime::ComputationClient::DataPtr> outputs =
            ShardingUtil::OutputHandler(
                runtime::GetComputationClient()->ExecuteReplicated(
                    *async->cached_computation->computation, device_arguments,
                    devices, execute_options),
                sharding_specs, /*replicated_output=*/false);
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

  runtime::env::ScheduleIoClosure(async->mwait.Completer(std::move(syncfn)));
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
  std::vector<XLATensor::ShardingSpecPtr> sharding_specs(coll->indices.size(),
                                                         nullptr);

  // Extract sharding specs for the results and prepare the sharded data
  // placeholders if the computation is sharded.
  if (cached_computation->is_sharded) {
    ShardingUtil::PrepareOutputShardingPropagation(
        tensors, coll->indices, cached_computation->computation, &tensors_data,
        &sharding_specs);
    DebugUtil::SaveOutputShardingInfo(tensors, coll->indices);
  }

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
        const xla::Shape& root_shape = ShapeHelper::ShapeOfXlaOp(root);
        auto parameter_data_shape =
            std::dynamic_pointer_cast<runtime::ComputationClient::Data>(
                parameters_data[i])
                ->shape();
        // Need to check whether existing buffer and the new value has the same
        // shape and the existing buffer has not been aliased before aliasing
        // the existing and new buffer.

        bool equal_sharding;
        // get sharding for the parameter data
        std::optional<xla::OpSharding> parameter_sharding =
            torch_xla::runtime::GetComputationClient()->GetDataSharding(
                std::dynamic_pointer_cast<runtime::ComputationClient::Data>(
                    parameters_data[i]));
        // get sharding for output tensor
        size_t output_tensor_index = indices[output_index];
        XLATensor::ShardingSpecPtr output_sharding =
            tensors[output_tensor_index]->sharding_spec();
        if (!parameter_sharding && !output_sharding) {
          // Both parameter and output does not have sharding.
          // TODO(JackCaoG): It is possible that output might get a sharding
          // after sharding propagation. Consier not aliased here(if under SPMD
          // mode).
          equal_sharding = true;
        } else if (parameter_sharding && output_sharding) {
          equal_sharding = ShardingUtil::EqualOpShardings(
              *parameter_sharding, output_sharding->sharding);
        } else {
          // one of the parameter and output does not have sharding.
          equal_sharding = false;
        }

        if (parameter_data_shape == root_shape && alias_map[output_index] < 0 &&
            equal_sharding) {
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
      runtime::sys_util::GetEnvBool("XLA_ENABLE_PARAM_ALIASING", true);
  static const size_t parameter_wrapping_threadshold =
      runtime::sys_util::GetEnvInt("XLA_PARAMETER_WRAPPING_THREADSHOLD", 3200);
  static const bool using_pjrt =
      runtime::sys_util::GetEnvString("PJRT_DEVICE", "").size() > 0;
  LoweringContext lowering_ctx("SyncTensorsGraph", coll.device,
                               po_data->post_order,
                               std::move(po_data->emission_map));
  for (auto ir_value : ir_values) {
    xla::XlaOp root = lowering_ctx.GetOutputOp(
        torch::lazy::Output(ir_value.node.get(), ir_value.index));
    lowering_ctx.AddResult(root);
  }
  // Always execute sharded when running in SPMD mode
  bool is_sharded = (coll.device == GetVirtualDevice());
  // Annotate HLO sharding selectively in the compuation.
  ShardingUtil::SetHloSharding(&lowering_ctx);

  std::vector<std::pair<int64_t, int64_t>> input_output_alias_pair;
  // TODO(yeounoh) aliasing is disabled for partitioned computation,
  // since the current aliasing compares the unpartitioned input and output
  // shapes which can lead to an incorrect aliasing pairs if sharded.
  if (enable_aliasing && coll.config.sync_ltc_data &&
      coll.config.force_ltc_data) {
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

  std::vector<runtime::ComputationClient::CompileInstance> instances;
  instances.push_back({std::move(computation), coll.device.toString(),
                       runtime::GetComputationClient()->GetCompilationDevices(
                           coll.device.toString(), devices),
                       &shape, should_wrap_parameter, is_sharded});

  TF_VLOG(3) << "Compiling IR graph hash "
             << torch::lazy::HashToString(coll.hash) << " on device "
             << coll.device << " ...";
  std::vector<std::shared_ptr<runtime::ComputationClient::Computation>>
      computations =
          runtime::GetComputationClient()->Compile(std::move(instances));
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
          computations.front(),
          /*parameters_data=*/std::move(po_data->parameters_data),
          /*is_sharded=*/is_sharded};
}

std::shared_ptr<XLAGraphExecutor::Async>
XLAGraphExecutor::SyncTensorsGraphInternal(
    std::vector<XLATensorPtr>* tensors, absl::Span<const std::string> devices,
    const SyncTensorsConfig& config, bool warm_up_cache_only) {
  tsl::profiler::TraceMe activity("SyncTensorsGraphInternal",
                                  tsl::profiler::TraceMeLevel::kInfo);
  if (runtime::sys_util::GetEnvBool("PT_XLA_DEBUG", false)) {
    DebugUtil::analyze_graph_execution_python_frame();
  }
  SyncTensorCollection coll = CollectSyncTensors(*tensors, config);
  if (coll.indices.empty()) {
    // Enure previous execution is complete before exiting this
    // function. Caller of `SyncTensorsGraphInternal` might want to call
    // wait() on the result of this function before accesing the value of
    // XLAData. If nullptr is returned here caller will assume there is no need
    // to wait. However in the cases of `SyncTensorsGraphInternal` being called
    // twice in a row, the first one will create placeholders then return,
    // second `SyncTensorsGraphInternal` will find there is nothing to sync and
    // return. It is possible that by the time second `SyncTensorsGraphInternal`
    // returned, first computation is still running. If user trying to call
    // `TransferFromServer` on placeholder XLAData, runtime will segfault. Force
    // the `SyncTensorsGraphInternal` to block until previous computation either
    // here or in `ScheduleSyncTensorsGraph` will solve this issue.
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
