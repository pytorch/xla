#include "torch_xla/csrc/xla_graph_executor.h"

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
#include "tensorflow/compiler/xla/xla_client/cache.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/env_vars.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/unique.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/lazy/core/hash.h"
#include "torch/csrc/lazy/core/helpers.h"
#include "torch/csrc/lazy/core/ir_util.h"
#include "torch/csrc/lazy/core/lazy_graph_executor.h"
#include "torch/csrc/lazy/core/metrics.h"
#include "torch/csrc/lazy/core/tensor_util.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/computation.h"
#include "torch_xla/csrc/debug_util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/layout_manager.h"
#include "torch_xla/csrc/op_by_op_executor.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/cast.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/ops/dynamic_ir.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/ops/view.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
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

// The DeviceContextArena holds per device live information and statistics,
// among which the XLA tensors which are currently alive in the system. This is
// used to create XLA computation "barriers" in order to flush pending
// operations and ensure the same XLA computations are created during the
// training loops.
class DeviceContextArena {
  struct DeviceContext {
    std::mutex lock;
    std::map<int64_t, std::weak_ptr<torch::lazy::LazyTensor::Data>> tensors_data;
    uint64_t seed = 101;
    uint64_t running_seed = 101;
    torch::lazy::Value seed_ir_value;
  };

 public:
  static DeviceContextArena* Get() {
    static DeviceContextArena* arena = new DeviceContextArena();
    return arena;
  }

  void RegisterTensor(std::shared_ptr<torch::lazy::LazyTensor::Data> data) {
    DeviceContext* devctx = GetDeviceContext(data->device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->tensors_data.emplace(data->unique_id, data);
    TORCH_LAZY_COUNTER("CreateXlaTensor", 1);
  }

  void UnregisterTensor(torch::lazy::LazyTensor::Data* data) {
    DeviceContext* devctx = GetDeviceContext(data->device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->tensors_data.erase(data->unique_id);
    TORCH_LAZY_COUNTER("DestroyXlaTensor", 1);
  }

  std::vector<XLATensorPtr> GetLiveTensors(
      const torch::lazy::BackendDevice* device) {
    std::vector<XLATensorPtr> tensors;
    auto fn = [&](DeviceContext* devctx) {
      std::lock_guard<std::mutex> lock(devctx->lock);
      for (auto& uid_wptr : devctx->tensors_data) {
        auto data = std::dynamic_pointer_cast<XLATensor::Data>(uid_wptr.second.lock());
        if (data != nullptr) {
          tensors.push_back(
              c10::make_intrusive<XLATensor>(XLATensor(std::move(data))));
        }
      }
    };
    ForAllDeviceContexts(fn, device);
    return tensors;
  }

  torch::lazy::Value GetRngSeed(const torch::lazy::BackendDevice& device) {
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

  torch::lazy::BackendDataPtr GetBaseSeedData(
      const torch::lazy::BackendDevice& device) {
    static const at::ScalarType kSeedType = at::ScalarType::Long;
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    at::Tensor tensor = at::scalar_tensor(MakeIntScalar(devctx->seed),
                                          at::TensorOptions(kSeedType));
    torch::lazy::BackendDataPtr device_data = TensorToXlaData(tensor, device);
    devctx->seed_ir_value = torch::lazy::MakeNode<DeviceData>(device_data);
    devctx->running_seed = devctx->seed;
    return torch_xla::DeviceData::Cast(devctx->seed_ir_value.node.get())
        ->data();
  }

  uint64_t GetRunningSeed(const torch::lazy::BackendDevice& device) {
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    return devctx->running_seed;
  }

  void SetRngSeed(const torch::lazy::BackendDevice& device, uint64_t seed) {
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->seed = seed;
    devctx->running_seed = devctx->seed;
    devctx->seed_ir_value = torch::lazy::Value();
  }

  void MarkStep(const torch::lazy::BackendDevice& device) {
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->seed = 1012031 + devctx->seed * 7012063;
    devctx->running_seed = devctx->seed;
    devctx->seed_ir_value = torch::lazy::Value();
  }

 private:
  std::vector<DeviceContext*> GetAllDeviceContexts() {
    std::vector<DeviceContext*> all_device_contexts;
    std::lock_guard<std::mutex> lock(lock_);
    all_device_contexts.reserve(device_contexts_.size());
    for (auto& device_contexts : device_contexts_) {
      all_device_contexts.push_back(device_contexts.second);
    }
    return all_device_contexts;
  }

  void ForAllDeviceContexts(const std::function<void(DeviceContext*)>& fn,
                            const torch::lazy::BackendDevice* device) {
    if (device == nullptr) {
      for (auto devctx : GetAllDeviceContexts()) {
        fn(devctx);
      }
    } else {
      fn(GetDeviceContext(*device));
    }
  }

  DeviceContext* GetDeviceContext(const torch::lazy::BackendDevice& device) {
    std::lock_guard<std::mutex> lock(lock_);
    auto it = device_contexts_.find(device);
    if (it == device_contexts_.end()) {
      it = device_contexts_.emplace(device, new DeviceContext()).first;
    }
    return it->second;
  }

  std::mutex lock_;
  std::map<torch::lazy::BackendDevice, DeviceContext*> device_contexts_;
};

XLAGraphExecutor::Async::Async(
    SyncTensorCollection* coll,
    std::vector<torch::lazy::BackendDataPtr> parameters_data,
    std::vector<torch::lazy::BackendDataPtr> tensors_data,
    ComputationCache::TypePtr cached_computation)
    : mwait(1),
      indices(std::move(coll->indices)),
      unlocker(std::move(coll->unlocker)),
      parameters_data(std::move(parameters_data)),
      device(coll->device.toString()),
      cached_computation(std::move(cached_computation)),
      tensors_data(std::move(tensors_data)) {}

void XLAGraphExecutor::Async::Wait() {
  mwait.Wait();
  // Accessing other Async members is safe only after MultiWait::Wait()
  // completes.
  torch::lazy::ExceptionCleanup::StatusType status;
  for (auto& cleanup : unlocker) {
    const torch::lazy::ExceptionCleanup::StatusType& cleanup_status =
        cleanup.GetStatus();
    if (cleanup_status != nullptr) {
      if (status == nullptr) {
        status = cleanup_status;
      }
      // If we observe the status here, no need to let it propagate to the next
      // device lock operation.
      cleanup.SetStatus(nullptr);
    }
  }
  if (status != nullptr) {
    std::rethrow_exception(status);
  }
}

XLAGraphExecutor* XLAGraphExecutor::Get() {
  static XLAGraphExecutor arena = XLAGraphExecutor();
  return &arena;
}

void XLAGraphExecutor::RegisterTensor(std::shared_ptr<XLATensor::Data> data) {
  DeviceContextArena::Get()->RegisterTensor(data);
}

void XLAGraphExecutor::UnregisterTensor(XLATensor::Data* data) {
  DeviceContextArena::Get()->UnregisterTensor(data);
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
                                        bool wait, bool sync_ltc_data) {
  TF_VLOG(4) << "Trying to sync the value of " << tensors->size()
             << " tensor(s)";
  tensorflow::profiler::TraceMe activity(
      "SyncTensorsGraph", tensorflow::profiler::TraceMeLevel::kInfo);
  static const bool op_by_op =
      xla::sys_util::GetEnvBool("XLA_SYNC_TENSORS_OPBYOP", false);
  SyncTensorsConfig config;
  config.sync_ltc_data = sync_ltc_data;
  if (op_by_op) {
    OpByOpAsync async = SyncTensorsGraphOpByOp(tensors, devices, config);
    if (wait) {
      async.Wait();
    }
  } else {
    auto async = SyncTensorsGraphInternal(tensors, devices, config);
    if (wait && async != nullptr) {
      async->mwait.Wait();
    }
  }
}

void XLAGraphExecutor::SyncLiveTensorsGraph(
    const torch::lazy::BackendDevice* device,
    absl::Span<const std::string> devices, bool wait) {
  tensorflow::profiler::TraceMe activity(
      "SyncLiveTensorsGraph", tensorflow::profiler::TraceMeLevel::kInfo);
  auto tensors = GetLiveTensors(device);
  TF_VLOG(4) << tensors.size() << " live tensors: devices=("
             << absl::StrJoin(devices, ",") << ")";
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

  SyncTensorCollection coll = CollectSyncTensors(tensors, config);
  absl::Span<const size_t> indices = coll.indices;
  std::vector<torch::lazy::Value> ir_values;
  ir_values.reserve(indices.size());
  for (auto index : indices) {
    ir_values.push_back(tensors.at(index)->CurrentIrValue());
  }
  PostOrderData po_data = RunPostOrder(ir_values, &coll);

  return torch::lazy::HashCombine(
      coll.hash, torch::lazy::Hash(po_data.parameter_sequence));
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
  tensorflow::profiler::TraceMe activity(
      "CollectSyncTensors", tensorflow::profiler::TraceMeLevel::kInfo);
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
  static const std::string invalid_device(
      "Unknown0"); /* Temp solution to identify unassigned devices */
  if (coll->device.toString().compare(invalid_device) == 0 ||
      coll->unlocker.size() > 0) {
    return;
  }
  // TODO(yeounoh) lock SPMD device
  coll->unlocker = DeviceLockerArena::Get()->LockDevices({coll->device});
}

std::vector<torch::lazy::BackendDataPtr>
XLAGraphExecutor::ExecuteComputationWithBarrier(
    torch::lazy::ComputationPtr computation,
    c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
    const torch::lazy::BackendDevice& device) {
  std::vector<torch::lazy::ExceptionCleanup> unlocker;
  unlocker = DeviceLockerArena::Get()->LockDevices({device});
  return torch::lazy::getBackend()->ExecuteComputation(computation, arguments,
                                                       device);
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
  std::vector<xla::Literal> literals =
      xla::ComputationClient::Get()->TransferFromServer(
          UnwrapXlaData(tensors_data));
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

std::vector<torch::lazy::BackendDataPtr> XLAGraphExecutor::SetTensorData(
    std::vector<XLATensorPtr>* tensors, const SyncTensorsConfig& config,
    absl::Span<const size_t> indices,
    const std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec) {
  tensorflow::profiler::TraceMe activity(
      "SetTensorData", tensorflow::profiler::TraceMeLevel::kInfo);
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
  tensorflow::profiler::TraceMe activity(
      "ExtractIRAndPrepareXlaData_", tensorflow::profiler::TraceMeLevel::kInfo);
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
    ComputationCache::TypePtr cached_computation) {
  tensorflow::profiler::TraceMe activity(
      "ScheduleSyncTensorsGraph", tensorflow::profiler::TraceMeLevel::kInfo);
  TensorCollectionBarrier(coll);
  std::shared_ptr<XLAGraphExecutor::Async> async = std::make_shared<Async>(
      coll, std::move(parameters_data), std::move(tensors_data),
      std::move(cached_computation));

  auto syncfn = [async, hash = coll->hash]() {
    try {
      std::vector<torch::lazy::BackendDataPtr> results;
      // Execute replicated if the compiled computation is partitioned.
      if (async->cached_computation->is_sharded) {
        // TODO(yeounoh) use local devices and verify with the pod execution.
        std::vector<std::string> devices =
            xla::ComputationClient::Get()->GetLocalDevices();
        std::vector<std::vector<xla::ComputationClient::DataPtr>>
            device_arguments = torch_xla::ShardingUtil::InputHandler(
                UnwrapXlaData(async->parameters_data), devices);
        xla::ComputationClient::ExecuteReplicatedOptions execute_options;
        TF_VLOG(3) << "Executing IR graph hash "
                   << torch::lazy::HashToString(hash)
                   << " on devices: " << absl::StrJoin(devices, ",");
        // TODO(jwtan): Remove the WrapXlaData when inherits LazyGraphExecutor.
        results = WrapXlaData(xla::ComputationClient::Get()->ExecuteReplicated(
            *async->cached_computation->computation->client_computation(),
            device_arguments, devices,
            execute_options)[0]);  // TODO(yeounoh) assumes replicated outputs
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
            ParseDeviceString(async->device));
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
  return ScheduleSyncTensorsGraph(coll, std::move(parameters_data),
                                  std::move(tensors_data),
                                  std::move(cached_computation));
}

XLAGraphExecutor::PostOrderData XLAGraphExecutor::RunPostOrder(
    const std::vector<torch::lazy::Value>& ir_values,
    SyncTensorCollection* coll) {
  tensorflow::profiler::TraceMe activity(
      "RunPostOrder", tensorflow::profiler::TraceMeLevel::kInfo);
  std::vector<const torch::lazy::Node*> roots;
  roots.reserve(ir_values.size());
  for (auto ir_value : ir_values) {
    roots.push_back(ir_value.node.get());
  }
  PostOrderData po_data;
  po_data.post_order =
      torch::lazy::Util::ComputePostOrder(roots, &po_data.emission_map);
  std::unordered_map<xla::ComputationClient::Data::OpaqueHandle, size_t>
      data_handles;

  for (auto node : po_data.post_order) {
    const auto backend_data =
        torch::lazy::getBackend()->GetComputationDataFromNode(node);
    if (backend_data != nullptr) {
      /* Acceptable race condition: HasValue may return false. This is OK
       * since the conditional barrier is a performance optimization. */
      if (!backend_data->HasValue()) {
        TensorCollectionBarrier(coll);
      }
      xla::ComputationClient::Data::OpaqueHandle handle =
          backend_data->GetHandle();
      auto it = data_handles.find(handle);
      if (it != data_handles.end()) {
        po_data.parameter_sequence.push_back(it->second);
      } else {
        po_data.parameter_sequence.push_back(po_data.parameters_data.size());
        data_handles[handle] = po_data.parameters_data.size();
        po_data.parameters_data.push_back(backend_data);
      }
    }
  }
  return po_data;
}

XLAGraphExecutor::ComputationCache::TypePtr
XLAGraphExecutor::LookupCachedCompile(const std::vector<XLATensorPtr>& tensors,
                                      const torch::lazy::hash_t& hash) {
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
      LookupCachedCompile(*tensors, coll->hash);
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
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t tensor_index = indices[i];
    int64_t tensor_id = tensors[tensor_index]->GetUniqueId();
    output_tensor_id_map[tensor_id] = i;
  }
  // TODO we need xla_shape here.
  const auto& parameters_data = lowering_ctx->GetParametersData();
  std::vector<ssize_t> alias_map(indices.size(), -1);
  for (size_t i = 0; i < parameters_data.size(); ++i) {
    auto* data_info =
        static_cast<torch::lazy::LazyGraphExecutor::DeviceDataInfo*>(
            parameters_data[i]->info());
    if (data_info != nullptr && !data_info->read_only) {
      auto it = output_tensor_id_map.find(data_info->tensor_id);
      if (it != output_tensor_id_map.end()) {
        size_t output_index = it->second;
        xla::XlaOp root = lowering_ctx->GetResult(output_index);
        const xla::Shape& root_shape = XlaHelpers::ShapeOfXlaOp(root);
        auto parameter_data_shape = UnwrapXlaData(parameters_data[i])->shape();
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
  TORCH_LAZY_VALUE_METRIC("InputOutputAliasCount", alias_map.size());
  return input_output_alias_pair;
}

XLAGraphExecutor::CompilationResult XLAGraphExecutor::Compile(
    const std::vector<XLATensorPtr>& tensors,
    absl::Span<const std::string> devices, const SyncTensorCollection& coll,
    PostOrderData* po_data, const std::vector<torch::lazy::Value>& ir_values) {
  tensorflow::profiler::TraceMe activity(
      [&] {
        return tensorflow::profiler::TraceMeEncode(
            "XLAGraphExecutor::Compile",
            {{"graph_hash", torch::lazy::HashToString(coll.hash)}});
      },
      tensorflow::profiler::TraceMeLevel::kInfo);
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
  if (enable_aliasing && coll.config.sync_ltc_data && !is_sharded) {
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
    const SyncTensorsConfig& config) {
  tensorflow::profiler::TraceMe activity(
      "SyncTensorsGraphInternal", tensorflow::profiler::TraceMeLevel::kInfo);
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

  return ScheduleSyncTensorsGraph(
      tensors, &coll, std::move(compile_result.parameters_data),
      compile_result.device.toString(), std::move(cached_computation),
      tensor_data_vec);
}

}  // namespace torch_xla
