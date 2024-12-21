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
#include "torch_xla/csrc/runtime/pjrt_computation_client.h"
#include "torch_xla/csrc/runtime/runtime.h"
#include "torch_xla/csrc/runtime/stablehlo_helper.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/runtime/xla_util.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/thread_pool.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/version.h"
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
  return torch_xla::MakeNode<DeviceData>(std::move(device_data));
}

bool ShouldSyncIrValue(const torch::lazy::Value& ir_value) {
  return ir_value->op() != xla_not_supported;
}

XLAGraphExecutor::ComputationCache* CreateComputationCache() {
  static const size_t kMaxCacheSize =
      runtime::sys_util::GetEnvInt("XLA_COMPILATION_CACHE_SIZE", 2048);
  static const bool readonlyPersistentCache =
      runtime::sys_util::GetEnvBool("XLA_PERSISTENT_CACHE_READ_ONLY", false);
  static std::string persistentCacheDir =
      runtime::sys_util::GetEnvString("XLA_PERSISTENT_CACHE_PATH", "");
  if (!persistentCacheDir.empty()) {
    auto serialize_fn =
        [](XLAGraphExecutor::ComputationCache::TypePtr computation)
        -> std::string {
      return runtime::GetComputationClient()->SerializeComputation(
          computation->computation);
    };
    auto deserialize_fn = [](std::string serialization)
        -> XLAGraphExecutor::ComputationCache::TypePtr {
      runtime::ComputationClient::ComputationPtr computation =
          runtime::GetComputationClient()->DeserializeComputation(
              serialization);
      if (!computation) return nullptr;
      return std::make_shared<XLAGraphExecutor::CachedComputation>(
          computation, /*is_sharded=*/UseVirtualDevice());
    };
    if (runtime::sys_util::GetEnvBool("XLA_HLO_DEBUG", false) ||
        runtime::sys_util::GetEnvBool("XLA_IR_DEBUG", false)) {
      TF_LOG(WARNING)
          << "Using persistent compilation cache with XLA_HLO_DEBUG=1 "
             "or XLA_IR_DEBUG=1 is not recommended. Changes to the HLO "
             "metadata will not be reflected in loaded executables.";
    }
    return new XLAGraphExecutor::PersistentCache(
        kMaxCacheSize, persistentCacheDir, readonlyPersistentCache,
        serialize_fn, deserialize_fn);
  }
  return new XLAGraphExecutor::MemoryCache(kMaxCacheSize);
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
  // Compose new seeds from the root seed, to avoid creating too many XLA
  // computation parameters which might overflow the TPU capacity.
  torch::lazy::Value k = ScalarOp(MakeIntScalar(kSeedMul),
                                  MakeXlaPrimitiveType(kSeedType, &device));
  torch::lazy::Value b = ScalarOp(MakeIntScalar(kSeedAdd),
                                  MakeXlaPrimitiveType(kSeedType, &device));
  if (XLAGraphExecutor::Get()->UseEagerMode()) {
    // In eager mode we want to make sure that `seed_ir_value` is always just
    // a device data instead a long sequence of pending IR.
    torch::lazy::Value seed_to_return = devctx->seed_ir_value;
    devctx->seed = kSeedAdd + kSeedMul * devctx->seed;
    devctx->running_seed = devctx->seed;
    // reset the `seed_ir_value`. Next time `seed_ir_value` will be generated
    // based on devctx->seed.
    devctx->seed_ir_value = torch::lazy::Value();
    return seed_to_return;
  } else {
    // Keep the running seed as scalar as well, so we can return it directly
    // without executing graphs.
    devctx->running_seed = kSeedAdd + kSeedMul * devctx->running_seed;
    devctx->seed_ir_value = b + k * devctx->seed_ir_value;
    return devctx->seed_ir_value;
  }
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
  devctx->seed_ir_value = torch_xla::MakeNode<DeviceData>(device_data);
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
    std::stringstream ss;
    ss << DebugUtil::GetTensorsGraphInfo(tensors, indices, format);
    ss << "Graph Hash: " << torch::lazy::HashToString(hash)
       << "\n\n## END_GRAPH\n\n";
    hash_to_graph_map[hash] = ss.str();
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
  return torch_xla::MakeNode<DeviceData>(std::move(device_data));
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
  return torch_xla::MakeNode<DeviceData>(std::move(data));
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
    ir_value = torch_xla::MakeNode<Expand>(
        ir_value, torch::lazy::ToVector<int64_t>(dimensions));
  }
  return ir_value;
}

torch::lazy::Value XLAGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value, xla::PrimitiveType type,
    c10::SymIntArrayRef sym_size, const torch::lazy::BackendDevice& device) {
  torch::lazy::Value ir_value = GetIrValueForScalar(value, type, device);
  SymIntElements size_elements = SymIntElements(sym_size);
  return torch_xla::MakeNode<ExpandSymInt>(ir_value, size_elements);
}

torch::lazy::Value XLAGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value, const xla::Shape& shape,
    const torch::lazy::BackendDevice& device) {
  return GetIrValueForScalar(value, shape.element_type(), shape.dimensions(),
                             device);
}

torch::lazy::Value XLAGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value, const xla::Shape& shape,
    std::optional<at::ScalarType> logical_element_type,
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
    std::optional<at::ScalarType> logical_element_type,
    const torch::lazy::BackendDevice& device) {
  xla::PrimitiveType primitive_type =
      logical_element_type
          ? MakeXlaPrimitiveType(*logical_element_type, &device)
          : shape.element_type();
  torch::lazy::Value ir_value =
      GetIrValueForScalar(value, primitive_type, device);
  return torch_xla::MakeNode<ExpandSymInt>(ir_value, size_elements);
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

void XLAGraphExecutor::SetAliasWithBufferDonorConfig(bool should_alias) {
  DeviceContextArena::Get()->SetAliasWithBufferDonorConfig(should_alias);
}

bool XLAGraphExecutor::GetAliasWithBufferDonorConfig() {
  return DeviceContextArena::Get()->GetAliasWithBufferDonorConfig();
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

void XLAGraphExecutor::MarkStep(const torch::lazy::BackendDevice& device,
                                bool reset_scope) {
  // TODO(jwtan): Replace this with TORCH_LAZY_COUNTER. We need MarkStep to
  // remain as XLA_COUNTER to support
  // runtime::metrics::CreatePerformanceReport(). For more information, see
  // NOTE: [TORCH_LAZY_COUNTER v.s. XLA_COUNTER].
  XLA_COUNTER("MarkStep", 1);
  DeviceContextArena::Get()->MarkStep(device);
  if (reset_scope) {
    torch::lazy::ScopePusher::ResetScopes();
  }
  ResetTrimCounter();
}

std::vector<size_t> GetBufferDonorIndexFromUserConfig(
    const std::vector<torch::lazy::BackendDataPtr>& parameters_data) {
  std::vector<size_t> buffer_donor_indexs;
  for (size_t i = 0; i < parameters_data.size(); ++i) {
    auto data = std::dynamic_pointer_cast<runtime::ComputationClient::Data>(
        parameters_data[i]);
    if (data->should_donate_buffer()) {
      buffer_donor_indexs.push_back(i);
    }
  }
  return buffer_donor_indexs;
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
  if (GetAliasWithBufferDonorConfig()) {
    std::vector<size_t> buffer_donor_index =
        GetBufferDonorIndexFromUserConfig(po_data.parameters_data);
    // Do not include hash on a empty vector.
    if (buffer_donor_index.size() > 0) {
      res_hash = torch::lazy::HashCombine(
          res_hash, torch::lazy::Hash(buffer_donor_index));
    }
  }
  {
    // Auto-sharding configs
    res_hash = torch::lazy::HashCombine(
        res_hash, torch::lazy::MHash(ShardingUtil::GetAutoSharding()));
    res_hash = torch::lazy::HashCombine(
        res_hash,
        torch::lazy::StringHash(
            runtime::sys_util::GetEnvString("XLA_AUTO_SPMD_MESH", "").c_str()));
  }
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

bool XLAGraphExecutor::IsComputationCacheInitialized() {
  return computation_cache_ != nullptr;
}

XLAGraphExecutor::ComputationCache* XLAGraphExecutor::GetComputationCache() {
  if (computation_cache_ == nullptr) {
    computation_cache_ = CreateComputationCache();
  }
  return computation_cache_;
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
        tensors[i]->data()->tensor_data = std::nullopt;
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
  // Ensure the compilation environment and git revision are reflected in the
  // hash.
  coll.hash = torch::lazy::HashCombine(
      coll.hash, runtime::GetComputationClient()->HashCompilationEnv());
  coll.hash =
      torch::lazy::HashCombine(coll.hash, torch::lazy::StringHash(XLA_GITREV));
  coll.config = config;
  coll.device = *unique_device;
  coll.indices.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    // Sync sharding annotations between the tensor and its node, if exists.
    // This either push down the sharding on the tensor to the IR before node
    // hash computation if the node has no annotation, or it actually syncs the
    // sharding attached to the node to the tensor, since sharding propagation &
    // resharding should attach the latest to the node.
    tensors[i]->sharding_spec();
    if (tensor_ids.insert(tensors[i]->GetUniqueId()).second &&
        // A tensor's xla_data might not be up to date if there is a view
        // associated with it. Make sure to sync those tensors here too.
        (tensors[i]->CurrentDataHandle() == nullptr ||
         (tensors[i]->data()->view != nullptr &&
          !tensors[i]->data()->view->IsUpToDate()))) {
      torch::lazy::Value ir_value = tensors[i]->CurrentIrValue();
      if (ir_value) {
        if (ShouldSyncIrValue(ir_value)) {
          auto device_data = torch_xla::DeviceData::Cast(ir_value.node.get());
          // If current tensor is cloned from another tensor, we want to assign
          // a new XlaData to it after current execution. Cloned tensor might
          // share the same storage with the origional tensor but origional
          // tensor might alias its storage with the output. It is safer to
          // allocate a new buffer for the cloned tensor.
          if (device_data != nullptr && !tensors[i]->data()->is_cloned) {
            // current IR is a devicedata, we don't need to include it as a
            // result of the computation. Call `GetXlaData` to extract the
            // XlaData from the DeviceData Node and reset the IR. We also want
            // to update XlaData's tensorID to make it match with the current
            // XLATensor.
            auto* data_info =
                static_cast<torch::lazy::LazyGraphExecutor::DeviceDataInfo*>(
                    device_data->data()->info());
            bool read_only = data_info != nullptr && data_info->read_only;
            tensors[i]->GetXlaData()->SetInfo(
                std::make_shared<LazyGraphExecutor::DeviceDataInfo>(
                    tensors[i]->GetUniqueId(), read_only));
          } else {
            // Add only tensors which need to be synced.
            coll.hash = torch::lazy::HashCombine(coll.hash, ir_value.hash());
            coll.indices.push_back(i);
          }
        }
      } else if (config.force_ltc_data) {
        // The tensor only has at::Tensor data. We need to queue it for a
        // device upload.
        std::optional<at::Tensor> tensor_data = tensors[i]->CurrentTensorData();
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

  DebugUtil::analyze_graph_execution_python_frame(
      DebugUtil::GraphAnalysisSource::DynamoExecution,
      /*graph_hash=*/hash,
      /*program_shape=*/&(cachedComputation->computation->program_shape()));

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
          *output_shapes, cachedComputation->computation);
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
        bool is_non_data_ir =
            xla_tensor_ptr->CurrentIrValue().node != nullptr &&
            (torch_xla::DeviceData::Cast(
                 xla_tensor_ptr->CurrentIrValue().node.get()) == nullptr);
        XLA_CHECK(!is_non_data_ir)
            << "input data to dynamo graph can not be a pending ir, please set "
               "`torch_xla._dynamo.config.skip_input_data_check` to False";
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
        // TODO(JackCaoG): handle eager mode
        std::vector<std::string> devices =
            runtime::GetComputationClient()->GetLocalDevices();
        runtime::ComputationClient::ExecuteReplicatedOptions execute_options;
        // OutputHandler creates sharded data for sharded
        // tensor results. Both sharded and unsharded results should be
        // "Assign"ed to the corresponding data placeholders.
        std::vector<runtime::ComputationClient::DataPtr> outputs =
            runtime::GetComputationClient()->ExecuteReplicated(
                *async->cached_computation->computation,
                UnwrapXlaData(async->parameters_data), devices,
                execute_options);
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

  thread::Schedule(async->mwait.Completer(std::move(syncfn)));

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
  ConvertStableHloToHlo(&mlir_module, &context, &hlo_proto);
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
      tensor->data()->tensor_data = std::nullopt;
      tensor->data()->is_cloned = false;
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
      std::optional<at::Tensor> tensor_data =
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
  DebugUtil::analyze_graph_execution_python_frame(
      DebugUtil::GraphAnalysisSource::Execution,
      /*graph_hash=*/coll->hash,
      /*program_shape=*/&(cached_computation->computation->program_shape()));
  tsl::profiler::TraceMe activity("ScheduleSyncTensorsGraph",
                                  tsl::profiler::TraceMeLevel::kInfo);
  TensorCollectionBarrier(coll);
  std::shared_ptr<XLAGraphExecutor::Async> async = std::make_shared<Async>(
      coll, std::move(parameters_data), std::move(tensors_data),
      std::move(cached_computation));
  auto syncfn = [async, hash = coll->hash, sharding_specs = sharding_specs,
                 use_eager_mode = UseEagerMode()]() {
    try {
      std::vector<torch::lazy::BackendDataPtr> results;
      // Execute replicated if the compiled computation is partitioned.
      if (async->cached_computation->is_sharded) {
        std::vector<std::string> devices =
            runtime::GetComputationClient()->GetLocalDevices();
        runtime::ComputationClient::ExecuteReplicatedOptions execute_options;
        TF_VLOG(3) << "Executing IR graph hash "
                   << torch::lazy::HashToString(hash)
                   << " on devices: " << absl::StrJoin(devices, ",");
        // OutputHandler creates sharded data for sharded
        // tensor results. Both sharded and unsharded results should be
        // "Assign"ed to the corresponding data placeholders.
        std::vector<runtime::ComputationClient::DataPtr> outputs =
            runtime::GetComputationClient()->ExecuteReplicated(
                *async->cached_computation->computation,
                UnwrapXlaData(async->parameters_data), devices,
                execute_options);
        results = WrapXlaData(outputs);
        TORCH_LAZY_COUNTER("ExecuteReplicated", 1);
        TF_VLOG(3) << "Executing IR graph hash "
                   << torch::lazy::HashToString(hash)
                   << " on devices: " << absl::StrJoin(devices, ",")
                   << " done!";
      } else {
        TF_VLOG(3) << "Executing IR graph hash "
                   << torch::lazy::HashToString(hash) << " on device "
                   << async->device << " ...";
        std::vector<runtime::ComputationClient::DataPtr> outputs =
            runtime::GetComputationClient()->ExecuteComputation(
                *async->cached_computation->computation,
                UnwrapXlaData(async->parameters_data), async->device.toString(),
                {/*explode_tuple=*/true,
                 /*eager_mode=*/use_eager_mode});
        results = WrapXlaData(outputs);
        TORCH_LAZY_COUNTER("ExecuteComputation", 1);
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

  thread::Schedule(async->mwait.Completer(std::move(syncfn)));
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

std::pair<bool, std::shared_ptr<XLAGraphExecutor::Async>>
XLAGraphExecutor::TryRunCachedSync(
    std::vector<XLATensorPtr>* tensors, SyncTensorCollection* coll,
    PostOrderData* po_data,
    const std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec,
    bool warm_up_cache_only) {
  ComputationCache::TypePtr cached_computation =
      LookupCachedCompile(coll->hash);
  bool cache_hit = false;
  if (cached_computation == nullptr) {
    return std::pair<bool, std::shared_ptr<XLAGraphExecutor::Async>>(cache_hit,
                                                                     nullptr);
  } else {
    cache_hit = true;
  }
  TORCH_LAZY_VALUE_METRIC("TensorsGraphSize", po_data->post_order.size());
  TF_VLOG(5) << "TensorsGraphSize=" << po_data->post_order.size();

  if (ShardingUtil::GetAutoSharding()) {
    // TODO(yeounoh) we may be able to update the cache to avoid this.
    // The current issue is that we are not properly updating the original
    // tensors to track the new sharded data after resharding.
    const xla::HloModuleProto& computation_proto =
        cached_computation->computation->computation().proto();
    ShardingUtil::ReshardParameters(computation_proto, tensors,
                                    &po_data->parameters_data,
                                    &po_data->post_order);
    TF_VLOG(5) << "Parameter sequence hash after resharding: "
               << torch::lazy::Hash(po_data->parameter_sequence);
  }

  // don't schedule the execution if the purpose of this SyncTensor is just to
  // warm up the cache.
  return std::pair<bool, std::shared_ptr<XLAGraphExecutor::Async>>(
      cache_hit, warm_up_cache_only
                     ? nullptr
                     : ScheduleSyncTensorsGraph(
                           tensors, coll, std::move(po_data->parameters_data),
                           coll->device.toString(),
                           std::move(cached_computation), tensor_data_vec));
}

std::vector<size_t> GetBufferDonorIndexForStepMarker(
    const std::vector<XLATensorPtr>& tensors, absl::Span<const size_t> indices,
    const std::vector<torch::lazy::BackendDataPtr>& parameters_data) {
  std::unordered_map<int64_t, size_t> output_tensor_id_map;
  std::vector<size_t> buffer_donor_indexs;
  // tensors[indices] represent all tensors that needs to be updated after
  // the execution. We can only alias the current buffer of these tensors
  // since those buffers are no longer needed after execution.
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t tensor_index = indices[i];
    int64_t tensor_id = tensors[tensor_index]->data()->alias_id;
    output_tensor_id_map[tensor_id] = i;
  }
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
        buffer_donor_indexs.push_back(i);
      }
    }
  }
  return buffer_donor_indexs;
}

std::vector<size_t> XLAGraphExecutor::GetBufferDonors(
    const std::vector<XLATensorPtr>& tensors, const SyncTensorCollection& coll,
    const std::vector<torch::lazy::BackendDataPtr>& parameters_data) {
  static const bool enable_aliasing =
      runtime::sys_util::GetEnvBool("XLA_ENABLE_PARAM_ALIASING", true);
  static const bool use_autosharding = ShardingUtil::GetAutoSharding();

  std::vector<size_t> buffer_donor_indices;
  // TODO(yeounoh) enable aliasing is disabled for partitioned computation,
  // since the current aliasing compares the unpartitioned input and output
  // shapes which can lead to an incorrect aliasing pairs if sharded.
  if (enable_aliasing && !use_autosharding) {
    if (coll.config.sync_ltc_data && coll.config.force_ltc_data) {
      // We can only alias at the step barrier, when force_ltc_data is true.
      // Consider the case:
      //   1. Tensor A(DEVICE_DATA)
      //   2. Tensor B = A + 0.9
      //   3. A += 0.4
      // If we activate aliasing for A's graph, and we do:
      //   print(A)
      //   print(A)
      // The first print will update DEVICE_DATA' with DEVICE_DATA+0.4, and the
      // second print will again update DEVICE_DATA" with DEVICE_DATA'+0.4,
      // which will lead to incorrect results. We cannot normally turn A's state
      // into DEVICE_DATA, as if any of the sources is a view, this will not
      // lead to correct results (as A's value taken at different times need to
      // reflect view source changes):
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
      buffer_donor_indices = GetBufferDonorIndexForStepMarker(
          tensors, coll.indices, parameters_data);
    } else if (GetAliasWithBufferDonorConfig()) {
      // only alias based on buffer donor if LTC can't auto infer the input
      // output aliasing.
      buffer_donor_indices = GetBufferDonorIndexFromUserConfig(parameters_data);
    }
  }
  return buffer_donor_indices;
}

void XLAGraphExecutor::SetBufferDonors(
    LoweringContext* lowering_ctx,
    const std::vector<size_t>& buffer_donor_indexs) {
  const std::vector<torch::lazy::BackendDataPtr>& parameters_data =
      lowering_ctx->GetParametersData();
  for (size_t i : buffer_donor_indexs) {
    lowering_ctx->builder()->AddBufferDonor(/*param_number=*/i,
                                            /*param_index=*/{});
  }
  TORCH_LAZY_VALUE_METRIC("InputOutputAliasCount", buffer_donor_indexs.size());
}

XLAGraphExecutor::CompilationResult XLAGraphExecutor::Compile(
    std::vector<XLATensorPtr>& tensors, absl::Span<const std::string> devices,
    const SyncTensorCollection& coll, PostOrderData* po_data,
    const std::vector<torch::lazy::Value>& ir_values,
    const std::vector<size_t>& buffer_donor_indices) {
  tsl::profiler::TraceMe activity(
      [&] {
        return tsl::profiler::TraceMeEncode(
            "XLAGraphExecutor::Compile",
            {{"graph_hash", torch::lazy::HashToString(coll.hash)}});
      },
      tsl::profiler::TraceMeLevel::kInfo);
  static const size_t parameter_wrapping_threadshold =
      runtime::sys_util::GetEnvInt("XLA_PARAMETER_WRAPPING_THREADSHOLD", 3200);
  static const bool use_autosharding = ShardingUtil::GetAutoSharding();
  std::string graph_name =
      (CurrentGraphName() != "") ? CurrentGraphName() : "SyncTensorsGraph";
  LoweringContext lowering_ctx(graph_name, coll.device, po_data->post_order,
                               std::move(po_data->emission_map));
  for (auto ir_value : ir_values) {
    xla::XlaOp root = lowering_ctx.GetOutputOp(
        torch::lazy::Output(ir_value.node.get(), ir_value.index));
    lowering_ctx.AddResult(root);
  }
  // Always execute sharded when running in SPMD mode
  bool is_sharded = (coll.device == GetVirtualDevice()) || UseVirtualDevice();
  // Annotate HLO sharding selectively in the compuation.
  ShardingUtil::SetHloSharding(&lowering_ctx);

  SetBufferDonors(&lowering_ctx, buffer_donor_indices);

  xla::XlaComputation computation = ConsumeValue(lowering_ctx.BuildXla());
  xla::ProgramShape program_shape = ConsumeValue(computation.GetProgramShape());

  // TODO(yeounoh) enable wrapping with auto-sharding.
  bool should_wrap_parameter =
      (program_shape.parameters_size() >= parameter_wrapping_threadshold) &&
      !use_autosharding;
  if (should_wrap_parameter) {
    TF_VLOG(3) << "Wrapping graph with " << program_shape.parameters_size()
               << " parameters. Threadshold = "
               << parameter_wrapping_threadshold;

    // trying to get all op shardings
    std::vector<xla::HloSharding> param_shardings;
    if (is_sharded) {
      param_shardings = XlaHelpers::ExtractInputShardings(computation);
    }

    computation = ConsumeValue(
        XlaHelpers::WrapXlaComputation(computation, program_shape.parameters(),
                                       param_shardings, buffer_donor_indices));
    program_shape = ConsumeValue(computation.GetProgramShape());
  }
  xla::Shape shape = MakeShapeWithDeviceLayout(
      program_shape.result(), static_cast<XlaDeviceType>(coll.device.type()));

  std::vector<runtime::ComputationClient::CompileInstance> instances;
  instances.push_back({std::move(computation), coll.device.toString(),
                       runtime::GetComputationClient()->GetCompilationDevices(
                           coll.device.toString(), devices),
                       &shape, should_wrap_parameter, is_sharded});
  instances.front().eager_mode = UseEagerMode();
  if (use_autosharding) {
    TF_VLOG(5) << "use_auto_spmd_partitioning is set.";
    TF_CHECK(is_sharded) << "Auto-sharding pass requires SPMD mode.";
    instances.front().use_auto_spmd_partitioning = use_autosharding;
    TORCH_LAZY_COUNTER("CompileWithAutoSharding", 1);

    // Apply XLA_AUTO_SPMD_MESH if it is set.
    // TODO(yeounoh) allow multi mesh exploration.
    std::vector<int64_t> auto_spmd_mesh_shape =
        ShardingUtil::GetAutoShardingMesh();
    std::vector<int64_t> auto_spmd_mesh_ids =
        ShardingUtil::GetAutoShardingMeshIds(
            instances.front().computation.proto());
    instances.front().auto_spmd_mesh_shape = auto_spmd_mesh_shape;
    instances.front().auto_spmd_mesh_ids = auto_spmd_mesh_ids;
    TF_VLOG(5) << "auto_spmd_mesh_shape={"
               << absl::StrJoin(auto_spmd_mesh_shape, ",") << "}\n"
               << "auto_spmd_mesh_ids={"
               << absl::StrJoin(auto_spmd_mesh_ids, ",") << "}";
  }

  DebugUtil::analyze_graph_execution_python_frame(
      DebugUtil::GraphAnalysisSource::Compilation,
      /*graph_hash=*/coll.hash, /*program_shape=*/&program_shape);

  TF_VLOG(3) << "Compiling IR graph hash "
             << torch::lazy::HashToString(coll.hash) << " on device "
             << coll.device << " ...";
  std::vector<std::shared_ptr<runtime::ComputationClient::Computation>>
      computations =
          runtime::GetComputationClient()->Compile(std::move(instances));
  DebugUtil::post_compilation_analysis(computations[0]);
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

  if (use_autosharding) {
    const xla::HloModuleProto& computation_proto =
        computations.front()->computation().proto();
    ShardingUtil::ReshardParameters(computation_proto, &tensors,
                                    &po_data->parameters_data,
                                    &po_data->post_order);
    TF_VLOG(5) << "Parameter sequence hash after resharding: "
               << torch::lazy::Hash(po_data->parameter_sequence);
  }

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
          /*computation=*/computations.front(),
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
    // Enure previous execution is complete before exiting this
    // function. Caller of `SyncTensorsGraphInternal` might want to call
    // wait() on the result of this function before accesing the value of
    // XLAData. If nullptr is returned here caller will assume there is no need
    // to wait. However in the cases of `SyncTensorsGraphInternal` being called
    // twice in a row, the first one will create placeholders then return,
    // second `SyncTensorsGraphInternal` will find there is nothing to sync and
    // return. It is possible that by the time second `SyncTensorsGraphInternal`
    // returned, first computation is still running. If user trying to call
    // `TransferFromDevice` on placeholder XLAData, runtime will segfault. Force
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

  std::vector<size_t> buffer_donor_indices =
      GetBufferDonors(*tensors, coll, po_data.parameters_data);
  if (buffer_donor_indices.size() > 0) {
    // Do not include hash on a empty vector.
    coll.hash = torch::lazy::HashCombine(
        coll.hash, torch::lazy::Hash(buffer_donor_indices));
  }
  {
    // Auto-sharding configs
    coll.hash = torch::lazy::HashCombine(
        coll.hash, torch::lazy::MHash(ShardingUtil::GetAutoSharding()));
    coll.hash = torch::lazy::HashCombine(
        coll.hash,
        torch::lazy::StringHash(
            runtime::sys_util::GetEnvString("XLA_AUTO_SPMD_MESH", "").c_str()));
  }

  DebugUtil::SaveGraphHash(coll.hash);
  TF_VLOG(4) << "Parameter sequence graph hash "
             << torch::lazy::HashToString(coll.hash);

  std::pair<bool, std::shared_ptr<XLAGraphExecutor::Async>> cache_res =
      TryRunCachedSync(tensors, &coll, &po_data, tensor_data_vec,
                       warm_up_cache_only);
  if (cache_res.first) {
    // we have a cache hit, execution has been scheduled by TryRunCachedSync.
    return cache_res.second;
  }
  CompilationResult compile_result = Compile(*tensors, devices, coll, &po_data,
                                             ir_values, buffer_donor_indices);

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
