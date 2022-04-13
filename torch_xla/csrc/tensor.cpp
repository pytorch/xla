#include "torch_xla/csrc/tensor.h"

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

#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/cache.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
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
#include "torch/csrc/lazy/core/tensor_util.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/debug_util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/layout_manager.h"
#include "torch_xla/csrc/op_by_op_executor.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/cast.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/ops/view.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace {

struct TlsData {
  void Reset() { trim_counter = 0; }

  size_t trim_counter = 0;
};

thread_local TlsData g_tls_data;

// Locking:
// We perform two kinds of operations of tensors, synchronous and asynchronous.
// The ApplyPendingGraph() are synchronous, as we need the device data result
// immediately. Before the synchronous operations can start, they need to wait
// that the pending asynchronous operations have completed.
// Synchronous operations do not hold device locks, since they are strictly
// sequential, dictated by the PyTorch execution order.
// The SyncTensorsGraph() is asynchronous, and returns immediately after having
// scheduled the asynchronous operation. While executing, the asynchronous
// operations will hold locks on all the participating devices (in most common
// cases there will be only one device).
// Since asynchronous operations capture device locks, only one asynchronous
// operation can execute at the same time, on a given device. Tensor operations
// which send data to device do not need to hold any device locks while doing
// so. Only operations which _use_ device data (computations, and transfer from
// server) need to wait for asynchronous operations to complete (barrier).

class DeviceLocker {
 public:
  explicit DeviceLocker(torch::lazy::BackendDevice device) : device_(std::move(device)) {}

  const torch::lazy::BackendDevice& device() const { return device_; }

  void Lock() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !locked_; });
    CheckResetException();
    locked_ = true;
  }

  void Unlock(std::exception_ptr exptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    locked_ = false;
    exptr_ = std::move(exptr);
    cv_.notify_all();
  }

  void Barrier() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !locked_; });
    cv_.notify_all();
    CheckResetException();
  }

 private:
  void CheckResetException() {
    std::exception_ptr exptr = std::move(exptr_);
    exptr_ = nullptr;
    if (exptr != nullptr) {
      std::rethrow_exception(exptr);
    }
  }

  torch::lazy::BackendDevice device_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool locked_ = false;
  std::exception_ptr exptr_;
};

class DeviceLockerArena {
 public:
  static DeviceLockerArena* Get() {
    static DeviceLockerArena* arena = new DeviceLockerArena();
    return arena;
  }

  std::shared_ptr<DeviceLocker> GetLocker(const torch::lazy::BackendDevice& device) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = lockers_.find(device);
    if (it == lockers_.end()) {
      it = lockers_.emplace(device, std::make_shared<DeviceLocker>(device))
               .first;
    }
    return it->second;
  }

 private:
  std::mutex mutex_;
  std::map<torch::lazy::BackendDevice, std::shared_ptr<DeviceLocker>> lockers_;
};

xla::util::ExceptionCleanup LockDevice(const torch::lazy::BackendDevice& device) {
  auto locker = DeviceLockerArena::Get()->GetLocker(device);
  locker->Lock();
  return xla::util::ExceptionCleanup(
      [locker =
           std::move(locker)](xla::util::ExceptionCleanup::StatusType status) {
        locker->Unlock(std::move(status));
      });
}

void DeviceBarrier(const torch::lazy::BackendDevice& device) {
  auto locker = DeviceLockerArena::Get()->GetLocker(device);
  locker->Barrier();
}

// Use a set to impose an order on the device locking sequence (ABBA
// prevention).
std::vector<xla::util::ExceptionCleanup> LockDevices(
    const std::set<torch::lazy::BackendDevice>& devices) {
  std::vector<xla::util::ExceptionCleanup> unlocker;
  unlocker.reserve(devices.size());
  for (auto& device : devices) {
    unlocker.emplace_back(LockDevice(device));
  }
  return unlocker;
}

class XlaDataCacheArena {
 public:
  struct TensorHasher {
    size_t operator()(const at::Tensor& tensor) const {
      return torch::lazy::HashReduce(torch::lazy::HashCombine(
          torch::lazy::GetEnumValue(tensor.scalar_type()), TensorHash(tensor)));
    };
  };
  struct TensorComparer {
    bool operator()(const at::Tensor& tensor1,
                    const at::Tensor& tensor2) const {
      return TensorCompare(tensor1, tensor2);
    }
  };

  using XlaDataCache =
      xla::util::Cache<at::Tensor, xla::ComputationClient::Data, TensorHasher,
                       TensorComparer>;

  explicit XlaDataCacheArena(size_t max_cache_size)
      : max_cache_size_(max_cache_size) {}

  XlaDataCache* Get(const torch::lazy::BackendDevice& device) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device_caches_.find(device);
    if (it == device_caches_.end()) {
      std::unique_ptr<XlaDataCache> cache(new XlaDataCache(max_cache_size_));
      it = device_caches_.emplace(device, std::move(cache)).first;
    }
    return it->second.get();
  }

 private:
  size_t max_cache_size_ = 0;
  std::mutex mutex_;
  std::map<torch::lazy::BackendDevice, std::unique_ptr<XlaDataCache>> device_caches_;
};

XlaDataCacheArena::XlaDataCache* GetXlaDataCache(const torch::lazy::BackendDevice& device) {
  static const size_t kMaxCacheSize =
      xla::sys_util::GetEnvInt("XLA_DEVDATA_CACHE_SIZE", 128);
  static XlaDataCacheArena* arena = new XlaDataCacheArena(kMaxCacheSize);
  return arena->Get(device);
}

ir::Value IrValueFromScalar(const at::Scalar& value, at::ScalarType scalar_type,
                            const torch::lazy::BackendDevice& device, bool transfer_async) {
  at::Tensor tensor = at::scalar_tensor(value, at::TensorOptions(scalar_type));
  xla::ComputationClient::DataPtr device_data =
      TensorToXlaData(tensor, device, transfer_async);
  return ir::MakeNode<ir::ops::DeviceData>(std::move(device_data));
}

xla::ComputationClient::DataPtr GetDeviceData(const at::Tensor& tensor,
                                              const torch::lazy::BackendDevice& device) {
  XlaDataCacheArena::XlaDataCache* cache = GetXlaDataCache(device);
  xla::ComputationClient::DataPtr device_data = cache->Get(tensor);
  if (device_data == nullptr) {
    at::Tensor tensor_copy = torch::lazy::CopyTensor(tensor);
    device_data = TensorToXlaData(tensor_copy, device);
    cache->Add(std::move(tensor_copy), device_data);
    XLA_COUNTER("DeviceDataCacheMiss", 1);
  }
  return device_data;
}

xla::ComputationClient::DataPtr GetDeviceData(const at::Scalar& value,
                                              at::ScalarType scalar_type,
                                              const torch::lazy::BackendDevice& device) {
  // Workaround since at::scalar_tensor doesn't support bfloat16 yet.
  at::Tensor t = at::scalar_tensor(
      value, at::TensorOptions(scalar_type == at::ScalarType::BFloat16
                                   ? at::ScalarType::Float
                                   : scalar_type));
  if (scalar_type == at::ScalarType::BFloat16) t = t.to(scalar_type);
  return GetDeviceData(t, device);
}

// Routing values to device data maximizes the changes for compilation cache
// hits, but it can prevent the compiler to perform optimizations. So tensor
// values which are within a given set, are routed to constant scalars if this
// API returns true.
bool IsSpecialScalar(const at::Scalar& value) {
  static bool no_scalars =
      xla::sys_util::GetEnvBool("XLA_NO_SPECIAL_SCALARS", false);
  if (!no_scalars && (value.isIntegral() || value.isFloatingPoint())) {
    double scalar_value = value.toDouble();
    return scalar_value == 0.0 || std::fabs(scalar_value) == 1.0;
  }
  return false;
}

bool ShouldSyncIrValue(const ir::Value& ir_value) {
  return ir_value->op() != ir::ops::xla_not_supported;
}

}  // namespace

// The DeviceContextArena holds per device live information and statistics,
// among which the XLA tensors which are currently alive in the system. This is
// used to create XLA computation "barriers" in order to flush pending
// operations and ensure the same XLA computations are created during the
// training loops.
class XLATensor::DeviceContextArena {
  struct DeviceContext {
    std::mutex lock;
    std::map<int64_t, std::weak_ptr<Data>> tensors_data;
    uint64_t seed = 101;
    uint64_t running_seed = 101;
    ir::Value seed_ir_value;
  };

 public:
  static DeviceContextArena* Get() {
    static DeviceContextArena* arena = new DeviceContextArena();
    return arena;
  }

  void RegisterTensor(std::shared_ptr<Data> data) {
    DeviceContext* devctx = GetDeviceContext(data->device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->tensors_data.emplace(data->unique_id, data);
    XLA_COUNTER("CreateXlaTensor", 1);
  }

  void UnregisterTensor(Data* data) {
    DeviceContext* devctx = GetDeviceContext(data->device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->tensors_data.erase(data->unique_id);
    XLA_COUNTER("DestroyXlaTensor", 1);
  }

  std::vector<XLATensor> GetLiveTensors(const torch::lazy::BackendDevice* device) {
    std::vector<XLATensor> tensors;
    auto fn = [&](DeviceContext* devctx) {
      std::lock_guard<std::mutex> lock(devctx->lock);
      for (auto& uid_wptr : devctx->tensors_data) {
        std::shared_ptr<Data> data = uid_wptr.second.lock();
        if (data != nullptr) {
          tensors.push_back(XLATensor(std::move(data)));
        }
      }
    };
    ForAllDeviceContexts(fn, device);
    return tensors;
  }

  ir::Value GetRngSeed(const torch::lazy::BackendDevice& device) {
    static const at::ScalarType kSeedType = at::ScalarType::Long;
    static const uint64_t kSeedMul = 214013;
    static const uint64_t kSeedAdd = 2531011;
    static bool transfer_async =
        xla::sys_util::GetEnvBool("XLA_TRANSFER_SEED_ASYNC", false);
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    if (!devctx->seed_ir_value) {
      devctx->seed_ir_value = IrValueFromScalar(
          MakeIntScalar(devctx->seed), kSeedType, device, transfer_async);
    }
    // Keep the running seed as scalar as well, so we can return it directly
    // without executing graphs.
    devctx->running_seed = kSeedAdd + kSeedMul * devctx->running_seed;
    // Compose new seeds from the root seed, to avoid creating too many XLA
    // computation parameters which might overflow the TPU capacity.
    ir::Value k = ir::ops::ScalarOp(MakeIntScalar(kSeedMul),
                                    MakeXlaPrimitiveType(kSeedType, &device));
    ir::Value b = ir::ops::ScalarOp(MakeIntScalar(kSeedAdd),
                                    MakeXlaPrimitiveType(kSeedType, &device));
    devctx->seed_ir_value = b + k * devctx->seed_ir_value;
    return devctx->seed_ir_value;
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
    devctx->seed_ir_value = ir::Value();
  }

  void MarkStep(const torch::lazy::BackendDevice& device) {
    DeviceContext* devctx = GetDeviceContext(device);
    std::lock_guard<std::mutex> lock(devctx->lock);
    devctx->seed = 1012031 + devctx->seed * 7012063;
    devctx->running_seed = devctx->seed;
    devctx->seed_ir_value = ir::Value();
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

struct DeviceDataInfo : public xla::ComputationClient::Data::Info {
  DeviceDataInfo(int64_t tensor_id, bool read_only)
      : tensor_id(tensor_id), read_only(read_only) {}

  int64_t tensor_id = 0;
  bool read_only = false;
};

XLATensor::Data::~Data() { DeviceContextArena::Get()->UnregisterTensor(this); }

XLATensor::Async::Async(
    SyncTensorCollection* coll,
    std::vector<xla::ComputationClient::DataPtr> parameters_data,
    std::vector<xla::ComputationClient::DataPtr> tensors_data,
    ComputationCache::TypePtr cached_computation)
    : mwait(1),
      indices(std::move(coll->indices)),
      unlocker(std::move(coll->unlocker)),
      parameters_data(std::move(parameters_data)),
      device(coll->device.toString()),
      cached_computation(std::move(cached_computation)),
      tensors_data(std::move(tensors_data)) {}

void XLATensor::Async::Wait() {
  mwait.Wait();
  // Accessing other Async members is safe only after MultiWait::Wait()
  // completes.
  xla::util::ExceptionCleanup::StatusType status;
  for (auto& cleanup : unlocker) {
    const xla::util::ExceptionCleanup::StatusType& cleanup_status =
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

XLATensor XLATensor::Create(const at::Tensor& tensor, const torch::lazy::BackendDevice& device) {
  XLA_CHECK_EQ(tensor.device().type(), at::kCPU);
  XLATensor xtensor(tensor, device);
  DeviceContextArena::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

XLATensor XLATensor::Create(
    xla::ComputationClient::DataPtr xla_data,
    c10::optional<at::ScalarType> logical_element_type) {
  XLATensor xtensor(std::move(xla_data), logical_element_type);
  DeviceContextArena::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

XLATensor XLATensor::Create(
    ir::Value ir_value, const torch::lazy::BackendDevice& device,
    c10::optional<at::ScalarType> logical_element_type) {
  XLATensor xtensor(std::move(ir_value), device, logical_element_type);
  DeviceContextArena::Get()->RegisterTensor(xtensor.data_ptr());
  if (UseEagerDebugMode()) {
    std::vector<XLATensor> xtensors({xtensor});
    ApplyEagerSync(xtensors);
  }
  return xtensor;
}

XLATensor XLATensor::Create(
    std::shared_ptr<View> view, const torch::lazy::BackendDevice& device,
    c10::optional<at::ScalarType> logical_element_type) {
  XLATensor xtensor(std::move(view), device, logical_element_type);
  DeviceContextArena::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

XLATensor::XLATensor(const at::Tensor& tensor, const torch::lazy::BackendDevice& device)
    : data_(std::make_shared<Data>(tensor, device)) {}

XLATensor::XLATensor(xla::ComputationClient::DataPtr xla_data,
                     c10::optional<at::ScalarType> logical_element_type)
    : data_(std::make_shared<Data>(xla_data, torch::lazy::BackendDevice(xla_data->device()),
                                   logical_element_type)) {}

XLATensor::XLATensor(ir::Value ir_value, const torch::lazy::BackendDevice& device,
                     c10::optional<at::ScalarType> logical_element_type)
    : data_(std::make_shared<Data>(std::move(ir_value), device,
                                   logical_element_type)) {
  TryLimitGraphSize();
}

XLATensor::XLATensor(std::shared_ptr<View> view, const torch::lazy::BackendDevice& device,
                     c10::optional<at::ScalarType> logical_element_type)
    : data_(std::make_shared<Data>(std::move(view), device,
                                   logical_element_type)) {}

XLATensor::XLATensor(std::shared_ptr<Data> data) : data_(std::move(data)) {}

XLATensor::Data* XLATensor::data() const {
  XLA_CHECK(data_ != nullptr) << "Trying to access a null cursor";
  return data_.get();
}

int64_t XLATensor::size(int64_t dim) const {
  auto xla_shape = shape();
  int rank = xla_shape.get().rank();
  int dim_index = torch::lazy::GetCanonicalDimensionIndex(dim, rank);
  return xla_shape.get().dimensions(dim_index);
}

at::ScalarType XLATensor::dtype() const {
  return data()->logical_element_type
             ? *data()->logical_element_type
             : TensorTypeFromXlaType(shape().get().element_type());
}

c10::optional<at::ScalarType> XLATensor::dtype_optional() const {
  return data()->logical_element_type;
}

xla::util::MaybeRef<xla::Shape> XLATensor::shape() const {
  if (data()->view != nullptr) {
    return data()->view->shape();
  }
  if (data()->xla_data != nullptr) {
    return data()->xla_data->shape();
  }
  if (data()->ir_value) {
    return data()->ir_value.xla_shape();
  }
  XLA_CHECK(data()->tensor_data);
  const torch::lazy::BackendDevice& device = GetDevice();
  return xla::ShapeUtil::MakeShape(
      MakeXlaPrimitiveType(data()->tensor_data->type().scalarType(), &device),
      XlaHelpers::I64List(data()->tensor_data->sizes()));
}

xla::Shape XLATensor::shape_with_layout() const {
  auto xla_shape = shape();
  return MakeArrayShapeFromDimensions(
      xla_shape.get().dimensions(), xla_shape.get().dynamic_dimensions(),
      xla_shape.get().element_type(), static_cast<XlaDeviceType>(GetDevice().type()));
}

const torch::lazy::BackendDevice& XLATensor::GetDevice() const { return data()->device; }

int64_t XLATensor::GetUniqueId() const { return data()->unique_id; }

std::ptrdiff_t XLATensor::GetViewAliasId() const {
  return data()->view != nullptr
             ? reinterpret_cast<std::ptrdiff_t>(data()->view->alias().get())
             : 0;
}

xla::ComputationClient::DataPtr XLATensor::GetXlaData() {
  // XLA data can coexist with a view, but we need to check that the view did
  // not receive any updates before calling the current XLA valid.
  bool up_to_date = true;
  ir::Value ir_value;
  if (data()->view != nullptr) {
    View::IrNode ir_value_updated = GetViewUpdate(data()->view);
    up_to_date = !ir_value_updated.updated;
    ir_value = std::move(ir_value_updated.ir_value);
  }
  if (up_to_date) {
    xla::ComputationClient::DataPtr xla_data = CurrentXlaData();
    if (xla_data != nullptr) {
      XLA_CHECK(xla_data->HasValue())
          << "Trying to access XLA data while an async operation is in flight: "
          << xla_data->shape();
      return xla_data;
    }
  }
  if (ir_value) {
    // The view gave us an updated IR value. We usually do not have a valid IR
    // value field together with a view, but to allow code reuse in
    // ApplyPendingGraph() we temporarily set it here. The following call to
    // ApplyPendingGraph() will clear it.
    AssignIrValue(std::move(ir_value));
  }
  if (data()->ir_value) {
    ApplyPendingGraph();
  } else {
    XLA_CHECK(data()->tensor_data);
    data()->xla_data = TensorToXlaData(*data()->tensor_data, GetDevice());
  }
  return data()->xla_data;
}

xla::ComputationClient::DataPtr XLATensor::CurrentXlaData() const {
  return data()->xla_data;
}

std::string XLATensor::DumpHloComputation(
    const std::vector<XLATensor>& tensors) {
  std::vector<ir::Value> ir_values;
  for (auto& tensor : tensors) {
    ir::Value ir_value = tensor.CurrentIrValue();
    if (ir_value) {
      ir_values.push_back(std::move(ir_value));
    }
  }
  return !ir_values.empty() ? ir::DumpUtil::ToHlo(ir_values, GetCurrentDevice())
                            : std::string();
}

void XLATensor::SetXlaData(xla::ComputationClient::DataPtr xla_data) {
  SetXlaData(std::move(xla_data), /*sync=*/true);
}

void XLATensor::SetXlaData(xla::ComputationClient::DataPtr xla_data,
                           bool sync) {
  data()->xla_data = std::move(xla_data);
  // Assigning a device data should always clear the IR node, to allow graph
  // trimming. A view cannot be reset though, unless we are at a step-end sync.
  AssignIrValue(ir::Value());
  if (sync) {
    data()->view = nullptr;
    data()->tensor_data = c10::nullopt;
  }
}

void XLATensor::SetIrValue(ir::Value ir_value, bool inplace) {
  data()->xla_data = nullptr;
  data()->tensor_data = c10::nullopt;
  if (data()->view != nullptr && inplace) {
    // If we have an active view, SetIrValue() happens, and we are
    // within an in-place execution context, we need to update the view's
    // alias as well.
    data()->view = UpdateView(data()->view, std::move(ir_value));
    data()->generation += 1;
  } else {
    // Reset the view if we are not within an in-place execution context
    data()->view = nullptr;
    data()->generation = 1;
    AssignIrValue(std::move(ir_value));
    TryLimitGraphSize();
  }
  if (UseEagerDebugMode() && ShouldSyncIrNode()) {
    std::vector<XLATensor> xtensors({*this});
    ApplyEagerSync(xtensors);
  }
}

void XLATensor::SetInPlaceIrValue(ir::Value ir_value) {
  auto xla_shape = shape();
  if (xla_shape.get().element_type() != ir_value.xla_shape().element_type()) {
    ir_value =
        ir::MakeNode<ir::ops::Cast>(ir_value, xla_shape.get().element_type());
  }
  SetIrValue(std::move(ir_value), /*inplace=*/true);
}

void XLATensor::AssignIrValue(ir::Value ir_value) const {
  data()->ir_value = std::move(ir_value);
  data()->generation += 1;
}

void XLATensor::TryLimitGraphSize() {
  static const size_t kCheckFrequency =
      xla::sys_util::GetEnvInt("XLA_TRIM_GRAPH_CHECK_FREQUENCY", 5000);
  static const size_t kMaxPendingGraphSize =
      xla::sys_util::GetEnvInt("XLA_TRIM_GRAPH_SIZE", 100000);
  if (data()->ir_value && ++g_tls_data.trim_counter % kCheckFrequency == 0) {
    size_t graph_size =
        torch::lazy::Util::GetGraphSize({data()->ir_value.node.get()});
    if (graph_size > kMaxPendingGraphSize) {
      XLA_COUNTER("TrimIrGraph", 1);
      ApplyPendingGraph();
    }
  }
}

ir::Value XLATensor::GetIrValue() const {
  ir::Value ir_value = CurrentIrValue();
  if (ir_value) {
    return ir_value;
  }
  xla::ComputationClient::DataPtr xla_data = CurrentXlaData();
  if (xla_data != nullptr) {
    // In case of tensor node, we do not clear the XLA data when we set the IR
    // node. This because we want further calls to GetIrValue() to fetch the
    // same IR node, and not create new ones (even though the lowering context
    // will still collapse them all into a single XLA parameter op). So call
    // which wants the XLA data will still find it, w/out having to fetch it via
    // a computation client from-server call.
    AssignIrValue(CreateTensorNode(xla_data, /*read_only=*/false));
    return data()->ir_value;
  }
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  XLA_CHECK(tensor_data);
  AssignIrValue(GetIrValueForTensor(*tensor_data, GetDevice()));
  return data()->ir_value;
}

ir::Value XLATensor::CurrentIrValue() const {
  if (data()->view != nullptr) {
    return GetViewUpdate(data()->view).ir_value;
  }
  return data()->ir_value;
}

void XLATensor::SetTensorData(at::Tensor tensor_data) {
  data()->tensor_data = std::move(tensor_data);
}

c10::optional<at::Tensor> XLATensor::CurrentTensorData() const {
  if (data()->view != nullptr && !data()->view->IsUpToDate()) {
    return c10::nullopt;
  }
  return data()->tensor_data;
}

ir::Value XLATensor::GetIrValueForTensor(const at::Tensor& tensor,
                                         const torch::lazy::BackendDevice& device) const {
  xla::ComputationClient::DataPtr data;
  bool read_only = false;
  if (tensor.dim() == 0 && tensor.numel() == 1) {
    at::Scalar value = tensor.item();
    if (IsSpecialScalar(value)) {
      return ir::ops::ScalarOp(
          std::move(value),
          MakeXlaPrimitiveType(tensor.scalar_type(), &device));
    }
    data = GetDeviceData(tensor, device);
    read_only = true;
  } else {
    XLA_TIMED("IrValueTensorToXlaData");
    data = TensorToXlaData(tensor, device);
  }
  return CreateTensorNode(std::move(data), read_only);
}

ir::Value XLATensor::GetDeviceDataIrValue(const at::Scalar& value,
                                          xla::PrimitiveType type,
                                          const torch::lazy::BackendDevice& device) {
  xla::ComputationClient::DataPtr data =
      GetDeviceData(value, TensorTypeFromXlaType(type), device);
  data->SetInfo(
      std::make_shared<DeviceDataInfo>(/*tensor_id=*/-1, /*read_only=*/true));
  return ir::MakeNode<ir::ops::DeviceData>(std::move(data));
}

ir::Value XLATensor::GetIrValueForConstant(const at::Scalar& value,
                                           const xla::Shape& shape) {
  ir::Value ir_value =
      ir::ops::ScalarOp(std::move(value), shape.element_type());
  if (!shape.dimensions().empty()) {
    ir_value = ir::MakeNode<ir::ops::Expand>(
        ir_value, torch::lazy::ToVector<int64_t>(shape.dimensions()));
  }
  return ir_value;
}

ir::Value XLATensor::GetIrValueForScalar(const at::Scalar& value,
                                         xla::PrimitiveType type,
                                         const torch::lazy::BackendDevice& device) {
  if (IsSpecialScalar(value)) {
    return ir::ops::ScalarOp(std::move(value), type);
  }
  return GetDeviceDataIrValue(value, type, device);
}

ir::Value XLATensor::GetIrValueForScalar(const at::Scalar& value,
                                         const torch::lazy::BackendDevice& device) {
  return GetIrValueForScalar(
      value, MakeXlaPrimitiveType(GetScalarType(value), &device), device);
}

ir::Value XLATensor::GetIrValueForScalar(const at::Scalar& value,
                                         xla::PrimitiveType type,
                                         absl::Span<const int64_t> dimensions,
                                         const torch::lazy::BackendDevice& device) {
  ir::Value ir_value = GetIrValueForScalar(value, type, device);
  if (!dimensions.empty()) {
    ir_value = ir::MakeNode<ir::ops::Expand>(
        ir_value, torch::lazy::ToVector<int64_t>(dimensions));
  }
  return ir_value;
}

ir::Value XLATensor::GetIrValueForScalar(const at::Scalar& value,
                                         const xla::Shape& shape,
                                         const torch::lazy::BackendDevice& device) {
  return GetIrValueForScalar(value, shape.element_type(), shape.dimensions(),
                             device);
}

ir::Value XLATensor::GetIrValueForScalar(
    const at::Scalar& value, const xla::Shape& shape,
    c10::optional<at::ScalarType> logical_element_type, const torch::lazy::BackendDevice& device) {
  xla::PrimitiveType type =
      logical_element_type
          ? MakeXlaPrimitiveType(*logical_element_type, &device)
          : shape.element_type();
  return GetIrValueForScalar(value, type, shape.dimensions(), device);
}

View::IrNode XLATensor::GetViewUpdate(const std::shared_ptr<View>& view) const {
  View::IrNode ir_value_updated = view->GetViewIrNode();
  if (ir_value_updated.updated) {
    data()->xla_data = nullptr;
    data()->tensor_data = c10::nullopt;
  }
  return ir_value_updated;
}

std::shared_ptr<View> XLATensor::UpdateView(std::shared_ptr<View> view,
                                            ir::Value ir_value) const {
  if (ir_value.xla_shape().dimensions() != view->shape().dimensions()) {
    XLA_CHECK_EQ(
        xla::util::Multiply<int64_t>(ir_value.xla_shape().dimensions()),
        xla::util::Multiply<int64_t>(view->shape().dimensions()));

    ViewInfo view_info(ViewInfo::Type::kReshape, ir_value.xla_shape(),
                       view->shape());
    view = view->CreateSubView(view_info.shape, view_info);
  }
  view->Update(std::move(ir_value));
  return view;
}

void XLATensor::SetSubView(ViewInfo view_info) const {
  data()->view = data()->view->CreateSubView(view_info.shape, view_info);
  data()->generation += 1;
}

void XLATensor::ModifyCurrentView(ViewInfo view_info) const {
  if (data()->view != nullptr) {
    SetSubView(view_info);
    return;
  }
  // This node is not a view. Since this function is meant to modify a view
  // in place, we need to turn this existing tensor into a view.
  ir::Value ir_value = GetIrValue();
  std::shared_ptr<Alias> alias = std::make_shared<Alias>(ir_value);
  data()->view =
      std::make_shared<View>(view_info.shape, alias, std::move(view_info));
  AssignIrValue(ir::Value());
}

std::shared_ptr<View> XLATensor::CreateView(ViewInfo view_info) const {
  if (data()->view != nullptr) {
    return data()->view->CreateSubView(view_info.shape, view_info);
  }
  // This node is not a view, and creating a view forks the current node into
  // becoming one itself. This means creating an alias with the current IR
  // Node, and using the same alias for the created IR Node.
  ir::Value ir_value = GetIrValue();
  std::shared_ptr<Alias> alias = std::make_shared<Alias>(ir_value);
  ViewInfo this_view_info(ViewInfo::Type::kNoOp, ir_value.xla_shape(),
                          ir_value.xla_shape());
  data()->view = std::make_shared<View>(ir_value.xla_shape(), alias,
                                        std::move(this_view_info));
  AssignIrValue(ir::Value());
  return std::make_shared<View>(view_info.shape, alias, view_info);
}

XLATensor XLATensor::CreateViewTensor(ViewInfo view_info) const {
  return Create(CreateView(std::move(view_info)), GetDevice(),
                dtype_optional());
}

at::Tensor XLATensor::ToTensor(bool detached) {
  at::Tensor tensor;
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  if (!tensor_data) {
    DeviceBarrier(GetDevice());
    // The GetXlaData() call will trigger an ApplyPendingGraph() if an IR Node
    // is available on the tensor.
    std::vector<at::Tensor> tensors = XlaDataToTensors({GetXlaData()}, dtype());
    tensor = std::move(tensors.front());
    if (!detached) {
      SetTensorData(tensor);
    }
  } else {
    tensor = *tensor_data;
    if (detached) {
      if (data()->ir_value || data()->xla_data != nullptr ||
          data()->view != nullptr) {
        // If we have other authoritive sources, just drop our reference and
        // transfer it to the caller.
        data()->tensor_data = c10::nullopt;
      } else {
        // Otherwise we need to make a copy to prevent the caller changing our
        // version.
        tensor = torch::lazy::CopyTensor(tensor);
      }
    }
  }
  return tensor;
}

void XLATensor::ShallowCopyTo(XLATensor* dest) const {
  dest->SetIrValue(GetIrValue(), /*inplace=*/false);
}

void XLATensor::SetScalarType(
    c10::optional<at::ScalarType> logical_element_type) {
  data()->logical_element_type = logical_element_type;
}

void XLATensor::SetTensor(at::Tensor tensor) {
  SetTensorData(tensor);
  data()->view = nullptr;
  data()->xla_data = nullptr;
  AssignIrValue(ir::Value());
}

void XLATensor::UpdateFromTensor(at::Tensor tensor, bool sync) {
  if (sync) {
    at::Tensor typed_tensor =
        torch::lazy::CopyTensor(tensor, dtype(), /*copy=*/false);
    SetIrValue(GetIrValueForTensor(typed_tensor, GetDevice()),
               /*inplace=*/true);
  } else {
    at::Tensor coyped_tensor = torch::lazy::CopyTensor(tensor, dtype());
    SetTensorData(coyped_tensor);
    data()->xla_data = nullptr;
    AssignIrValue(ir::Value());
    if (data()->view != nullptr) {
      ir::Value ir_value = GetIrValueForTensor(coyped_tensor, GetDevice());
      data()->view = UpdateView(data()->view, std::move(ir_value));
    }
  }
}

void XLATensor::UpdateFromTensorOut(at::Tensor tensor) {
  if (data()->view != nullptr &&
      xla::ShapeUtil::ElementsIn(shape()) != tensor.numel()) {
    data()->view = nullptr;
  }
  UpdateFromTensor(std::move(tensor), /*sync=*/false);
}

void XLATensor::UpdateFromTensorOut(const XLATensor& tensor) {
  if (data()->view != nullptr &&
      xla::ShapeUtil::ElementsIn(shape()) !=
          xla::ShapeUtil::ElementsIn(tensor.shape())) {
    data()->view = nullptr;
  }
  SetIrValue(tensor.GetIrValue(), /*inplace=*/true);
}

std::vector<XLATensor> XLATensor::GetLiveTensors(const torch::lazy::BackendDevice* device) {
  return DeviceContextArena::Get()->GetLiveTensors(device);
}

std::vector<xla::ComputationClient::DataPtr> XLATensor::GatherTensorsXlaData(
    const std::vector<XLATensor>& tensors, absl::Span<const size_t> indices,
    absl::Span<const xla::ComputationClient::DataPtr> tensors_data) {
  std::vector<xla::ComputationClient::DataPtr> result_tensors_data;
  std::unordered_map<int64_t, size_t> uid_index_map;
  size_t indices_index = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    int64_t tensor_id = tensors[i].GetUniqueId();
    auto it = uid_index_map.find(tensor_id);
    if (it != uid_index_map.end()) {
      // Current tensor is a duplicate of a previously processed tensor that had
      // an IR Node to sync. Get the XLA data from the tensor_data_map.
      result_tensors_data.push_back(result_tensors_data[it->second]);
    } else if (indices_index < indices.size() && i == indices[indices_index]) {
      // If we are at the current index (it means that the tensor at index
      // 'i' had an IR node to sync), use the XLA data held within the Async
      // object.
      uid_index_map.emplace(tensor_id, result_tensors_data.size());
      result_tensors_data.push_back(tensors_data[indices_index]);
      ++indices_index;
    } else if (!tensors[i].CurrentTensorData()) {
      xla::ComputationClient::DataPtr xla_data = tensors[i].CurrentXlaData();
      XLA_CHECK(xla_data != nullptr);
      result_tensors_data.push_back(std::move(xla_data));
    }
  }
  return result_tensors_data;
}

std::vector<at::Tensor> XLATensor::GetTensorsOpByOp(
    std::vector<XLATensor>* tensors) {
  SyncTensorsConfig config;
  config.force_xla_data = false;
  SyncTensorCollection coll = CollectSyncTensors(*tensors, config);
  std::vector<xla::ComputationClient::DataPtr> async_tensors_data;
  if (!coll.indices.empty()) {
    DebugUtil::SaveTensorsGraphInfo("GetTensorsOpByOp", *tensors,
                                    &coll.indices);

    std::vector<ir::Value> roots = CollectRoots(*tensors, coll.indices);
    async_tensors_data =
        OpByOpExecutor::Get()->Execute(roots, coll.device.toString(), {});
  }

  std::vector<xla::ComputationClient::DataPtr> tensors_data =
      GatherTensorsXlaData(*tensors, coll.indices, async_tensors_data);
  std::vector<xla::Literal> literals =
      xla::ComputationClient::Get()->TransferFromServer(tensors_data);

  return FetchTensors(tensors, literals, &coll.indices);
}

std::vector<at::Tensor> XLATensor::GetTensors(std::vector<XLATensor>* tensors) {
  TF_VLOG(4) << "Trying to get the value of " << tensors->size()
             << " tensor(s)";
  static const bool op_by_op =
      xla::sys_util::GetEnvBool("XLA_GET_TENSORS_OPBYOP", false);
  return op_by_op ? GetTensorsOpByOp(tensors) : GetTensorsFused(tensors);
}

std::vector<at::Tensor> XLATensor::GetTensorsFused(
    std::vector<XLATensor>* tensors) {
  SyncTensorsConfig config;
  config.force_xla_data = false;
  auto async = SyncTensorsGraphInternal(tensors, {}, config);
  if (async != nullptr) {
    async->mwait.Wait();
  }
  std::vector<xla::ComputationClient::DataPtr> tensors_data =
      GatherTensorsXlaData(
          *tensors,
          async != nullptr ? async->indices : absl::Span<const size_t>(),
          async != nullptr
              ? async->tensors_data
              : absl::Span<const xla::ComputationClient::DataPtr>());
  std::vector<xla::Literal> literals =
      xla::ComputationClient::Get()->TransferFromServer(tensors_data);
  return FetchTensors(tensors, literals,
                      async != nullptr ? &async->indices : nullptr);
}

std::vector<at::Tensor> XLATensor::FetchTensors(
    std::vector<XLATensor>* tensors, absl::Span<const xla::Literal> literals,
    const std::vector<size_t>* indices) {
  std::vector<at::Tensor> results;
  size_t literals_index = 0;
  size_t sync_index = 0;
  results.reserve(tensors->size());
  for (size_t i = 0; i < tensors->size(); ++i) {
    if (indices != nullptr && sync_index < indices->size() &&
        i == (*indices)[sync_index]) {
      results.push_back(MakeTensorFromXlaLiteral(literals[literals_index],
                                                 (*tensors)[i].dtype()));
      ++literals_index;
      ++sync_index;
    } else {
      c10::optional<at::Tensor> tensor_data = (*tensors)[i].CurrentTensorData();
      if (tensor_data) {
        results.push_back(*tensor_data);
      } else {
        XLA_CHECK_LT(literals_index, literals.size());
        results.push_back(MakeTensorFromXlaLiteral(literals[literals_index],
                                                   (*tensors)[i].dtype()));
        ++literals_index;
      }
    }
  }
  return results;
}

std::vector<XLATensor> XLATensor::CreateTensors(
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::string>& devices) {
  std::vector<xla::ComputationClient::DataPtr> handles =
      CreateTensorsData(tensors, devices);
  std::vector<XLATensor> xla_tensors;
  for (size_t i = 0; i < handles.size(); ++i) {
    xla_tensors.push_back(
        Create(std::move(handles[i]), tensors[i].scalar_type()));
  }
  return xla_tensors;
}

ir::Value XLATensor::CreateTensorNode(xla::ComputationClient::DataPtr data,
                                      bool read_only) const {
  data->SetInfo(std::make_shared<DeviceDataInfo>(GetUniqueId(), read_only));
  return ir::MakeNode<ir::ops::DeviceData>(std::move(data));
}

std::vector<XLATensor> XLATensor::MakeOutputTensors(
    torch::lazy::NodePtr node, bool inherit_logical_type) const {
  std::vector<XLATensor> tensors;
  tensors.reserve(node->num_outputs());
  for (size_t i = 0; i < node->num_outputs(); ++i) {
    if (inherit_logical_type) {
      tensors.push_back(CreateFrom(ir::Value(node, i)));
    } else {
      tensors.push_back(CreateFrom(ir::Value(node, i),
                                   /*logical_element_type=*/c10::nullopt));
    }
  }
  return tensors;
}

XLATensor XLATensor::CopyTensorToDevice(const torch::lazy::BackendDevice& device) {
  // TODO: This can be optimized via proper XRT/XLA computation.
  return Create(ToTensor(/*detached=*/true), device);
}

ir::Value XLATensor::MaybeCastIrValue(
    ir::Value ir_value, const torch::lazy::BackendDevice& device,
    c10::optional<at::ScalarType> logical_element_type) const {
  if (!logical_element_type) {
    logical_element_type = dtype_optional();
  }
  if (logical_element_type &&
      RequiresRawTypeCasting(*logical_element_type, &device)) {
    ir_value = ir::MakeNode<ir::ops::Cast>(ir_value, *logical_element_type);
  }
  return ir_value;
}

XLATensor XLATensor::CreateFrom(ir::Value ir_value) const {
  ir_value = MaybeCastIrValue(std::move(ir_value), GetDevice(),
                              /*logical_element_type=*/c10::nullopt);
  return Create(std::move(ir_value), GetDevice(), dtype_optional());
}

XLATensor XLATensor::CreateFrom(ir::Value ir_value,
                                const torch::lazy::BackendDevice& device) const {
  ir_value = MaybeCastIrValue(std::move(ir_value), device,
                              /*logical_element_type=*/c10::nullopt);
  return Create(std::move(ir_value), device, dtype_optional());
}

XLATensor XLATensor::CreateFrom(ir::Value ir_value,
                                at::ScalarType logical_element_type) const {
  ir_value =
      MaybeCastIrValue(std::move(ir_value), GetDevice(), logical_element_type);
  return Create(std::move(ir_value), GetDevice(), logical_element_type);
}

XLATensor XLATensor::CreateFrom(
    ir::Value ir_value,
    c10::optional<at::ScalarType> logical_element_type_opt) const {
  ir_value = MaybeCastIrValue(std::move(ir_value), GetDevice(),
                              logical_element_type_opt);
  return Create(std::move(ir_value), GetDevice(), logical_element_type_opt);
}

XLATensor XLATensor::CreateFrom(ir::Value ir_value, const torch::lazy::BackendDevice& device,
                                at::ScalarType logical_element_type) const {
  ir_value =
      MaybeCastIrValue(std::move(ir_value), device, logical_element_type);
  return Create(std::move(ir_value), device, logical_element_type);
}

void XLATensor::ApplyPendingGraph() {
  DeviceBarrier(GetDevice());
  // This method is called to ensure that the tensor data is available on
  // device, so that a call to CurrentXlaData() returns a valid pointer.
  if (CurrentXlaData() == nullptr) {
    std::vector<XLATensor> tensors({*this});
    SyncTensorsGraph(&tensors, {}, /*wait=*/true, /*sync_xla_data=*/false);
  }
}

void XLATensor::ApplyEagerSync(std::vector<XLATensor>& tensors) {
  SyncTensorsGraph(&tensors, {}, /*wait=*/false, /*sync_xla_data=*/false);
}

XLATensor::SyncTensorCollection XLATensor::CollectSyncTensors(
    const std::vector<XLATensor>& tensors, const SyncTensorsConfig& config) {
  tensorflow::profiler::TraceMe activity(
      "CollectSyncTensors", tensorflow::profiler::TraceMeLevel::kInfo);
  xla::util::Unique<torch::lazy::BackendDevice> unique_device;
  for (size_t i = 0; i < tensors.size(); ++i) {
    unique_device.set(tensors[i].GetDevice());
  }
  SyncTensorCollection coll;
  if (!unique_device) {
    return coll;
  }

  std::vector<at::Tensor> at_tensors;
  std::vector<std::string> devices;
  std::vector<size_t> at_tensor_index;
  std::unordered_set<int64_t> tensor_ids;
  // The force_xla_data controls aliasing compilation, so effectively the same
  // graph with on/off force_xla_data should not match, hash wise.
  coll.hash = torch::lazy::MHash(config.force_xla_data);
  coll.config = config;
  coll.device = *unique_device;
  coll.indices.reserve(tensors.size());
  TF_VLOG(4) << "Waiting on device barrier for device " << coll.device
             << " ...";
  {
    XLA_TIMED("DeviceLockWait");
    coll.unlocker = LockDevices(unique_device.AsSet());
  }
  TF_VLOG(4) << "Waiting on device barrier for device " << coll.device
             << " done!";
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensor_ids.insert(tensors[i].GetUniqueId()).second &&
        tensors[i].CurrentXlaData() == nullptr) {
      ir::Value ir_value = tensors[i].CurrentIrValue();
      if (ir_value) {
        if (ShouldSyncIrValue(ir_value)) {
          // Add only tensors which need to be synced.
          coll.hash = torch::lazy::HashCombine(coll.hash, ir_value.hash());
          coll.indices.push_back(i);
        }
      } else if (config.force_xla_data) {
        // The tensor only has at::Tensor data. We need to queue it for a
        // device upload.
        c10::optional<at::Tensor> tensor_data = tensors[i].CurrentTensorData();
        XLA_CHECK(tensor_data);
        at_tensors.push_back(*tensor_data);
        devices.push_back(tensors[i].GetDevice().toString());
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
    XLA_COUNTER("SyncTensorsToData", at_tensors.size());
    std::vector<xla::ComputationClient::DataPtr> handles =
        CreateTensorsData(at_tensors, devices);
    for (size_t i = 0; i < handles.size(); ++i) {
      // If we are here, it means that the IR Value for the tensor is not
      // present. Also, we uploaded the at::Tensor data to the device, but such
      // data is still valid so we leave it live on the XLA tensor (so that a
      // following ToTensor() does not need to fetch it from device).
      tensors[at_tensor_index[i]].data()->xla_data = std::move(handles[i]);
    }
  }
  TF_VLOG(4) << "Tensors graph hash " << torch::lazy::HashToString(coll.hash)
             << " on device " << coll.device;
  return coll;
}

XLATensor::ComputationCache::TypePtr XLATensor::LookupCachedCompile(
    const std::vector<XLATensor>& tensors, const torch::lazy::hash_t& hash) {
  ComputationCache::TypePtr cached_computation =
      GetComputationCache()->Get(hash);
  if (cached_computation == nullptr) {
    XLA_COUNTER("UncachedCompile", 1);
    return nullptr;
  }
  TF_VLOG(5) << "Graph hash " << torch::lazy::HashToString(hash)
             << " is computation hash "
             << torch::lazy::HashToString(torch::lazy::Hash(
                    cached_computation->computation->computation()
                        .proto()
                        .SerializeAsString()));
  XLA_COUNTER("CachedCompile", 1);
  return cached_computation;
}

std::shared_ptr<XLATensor::Async> XLATensor::TryRunCachedSync(
    std::vector<XLATensor>* tensors, SyncTensorCollection* coll,
    PostOrderData* po_data) {
  ComputationCache::TypePtr cached_computation =
      LookupCachedCompile(*tensors, coll->hash);
  if (cached_computation == nullptr) {
    return nullptr;
  }
  XLA_VALUE_METRIC("TensorsGraphSize", po_data->post_order.size());
  TF_VLOG(5) << "TensorsGraphSize=" << po_data->post_order.size();

  return ScheduleSyncTensorsGraph(
      tensors, coll, std::move(po_data->parameters_data),
      coll->device.toString(), std::move(cached_computation));
}

XLATensor::ComputationCache* XLATensor::GetComputationCache() {
  static const size_t kMaxCacheSize =
      xla::sys_util::GetEnvInt("XLA_COMPILATION_CACHE_SIZE", 1024);
  static ComputationCache* cache = new ComputationCache(kMaxCacheSize);
  return cache;
}

XLATensor::PostOrderData XLATensor::RunPostOrder(
    const std::vector<XLATensor>& tensors, absl::Span<const size_t> indices) {
  std::vector<const torch::lazy::Node*> roots;
  roots.reserve(indices.size());
  for (auto index : indices) {
    ir::Value ir_value = tensors.at(index).CurrentIrValue();
    roots.push_back(ir_value.node.get());
  }
  PostOrderData po_data;
  po_data.post_order = ir::Util::ComputePostOrder(roots, &po_data.emission_map);
  std::unordered_map<xla::ComputationClient::Data::OpaqueHandle, size_t>
      data_handles;
  for (auto node : po_data.post_order) {
    const ir::ops::DeviceData* device_data = ir::ops::DeviceData::Cast(node);
    if (device_data != nullptr) {
      xla::ComputationClient::Data::OpaqueHandle handle =
          device_data->data()->GetOpaqueHandle();
      auto it = data_handles.find(handle);
      if (it != data_handles.end()) {
        po_data.parameter_sequence.push_back(it->second);
      } else {
        po_data.parameter_sequence.push_back(po_data.parameters_data.size());
        data_handles[handle] = po_data.parameters_data.size();
        po_data.parameters_data.push_back(device_data->data());
      }
    }
  }
  return po_data;
}

std::vector<ir::Value> XLATensor::CollectRoots(
    const std::vector<XLATensor>& tensors, absl::Span<const size_t> indices) {
  std::vector<ir::Value> roots;
  roots.reserve(indices.size());
  for (auto index : indices) {
    roots.push_back(tensors.at(index).CurrentIrValue());
  }
  return roots;
}

std::vector<xla::ComputationClient::DataPtr> XLATensor::FetchTensorData(
    std::vector<XLATensor>* tensors, const SyncTensorsConfig& config,
    absl::Span<const size_t> indices) {
  std::vector<xla::ComputationClient::DataPtr> tensors_data;
  tensors_data.reserve(indices.size());
  for (auto index : indices) {
    XLATensor& tensor = (*tensors)[index];
    // If the config.force_xla_data flag is true, the purpose of this tensor
    // sync operation is to truncate the IR graph and materialize device data in
    // place of IR graph, on selected tensors. But since operation will complete
    // asynchronously, if a tensor does not already have device data, we need to
    // install a placeholder. Since at this point we hold a lock on the device
    // where the tensors reside (locks held within the coll structure, and moved
    // into the async variable), any other operation trying to access the
    // tensor's device data will have to wait until the asynchronous operation
    // completes.
    xla::ComputationClient::DataPtr xla_data = tensor.CurrentXlaData();
    if (xla_data == nullptr && config.force_xla_data) {
      const torch::lazy::BackendDevice& tensor_device = tensor.GetDevice();
      xla::Shape shape = MakeShapeWithDeviceLayout(
          tensor.shape(), static_cast<XlaDeviceType>(tensor_device.type()));
      xla_data = xla::ComputationClient::Get()->CreateDataPlaceholder(
          tensor_device.toString(), std::move(shape));
      tensor.SetXlaData(xla_data, config.sync_xla_data);
    }
    tensors_data.emplace_back(std::move(xla_data));
  }
  return tensors_data;
}

std::shared_ptr<XLATensor::Async> XLATensor::ScheduleSyncTensorsGraph(
    SyncTensorCollection* coll,
    std::vector<xla::ComputationClient::DataPtr> parameters_data,
    std::vector<xla::ComputationClient::DataPtr> tensors_data,
    ComputationCache::TypePtr cached_computation) {
  tensorflow::profiler::TraceMe activity(
      "ScheduleSyncTensorsGraph", tensorflow::profiler::TraceMeLevel::kInfo);
  std::shared_ptr<Async> async = std::make_shared<Async>(
      coll, std::move(parameters_data), std::move(tensors_data),
      std::move(cached_computation));

  auto syncfn = [async, hash = coll->hash]() {
    xla::ComputationClient::ExecuteComputationOptions options;
    try {
      TF_VLOG(3) << "Executing IR graph hash "
                 << torch::lazy::HashToString(hash) << " on device "
                 << async->device << " ...";
      auto results = xla::ComputationClient::Get()->ExecuteComputation(
          *async->cached_computation->computation, async->parameters_data,
          async->device, options);
      TF_VLOG(3) << "Executing IR graph hash "
                 << torch::lazy::HashToString(hash) << " on device "
                 << async->device << " done!";

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
      std::exception_ptr exptr = std::current_exception();
      for (auto& unlocker : async->unlocker) {
        unlocker.SetStatus(exptr);
      }
      throw;
    }
  };

  xla::env::ScheduleIoClosure(async->mwait.Completer(std::move(syncfn)));
  return async;
}

std::shared_ptr<XLATensor::Async> XLATensor::ScheduleSyncTensorsGraph(
    std::vector<XLATensor>* tensors, SyncTensorCollection* coll,
    std::vector<xla::ComputationClient::DataPtr> parameters_data,
    std::string device, ComputationCache::TypePtr cached_computation) {
  auto tensors_data = FetchTensorData(tensors, coll->config, coll->indices);
  return ScheduleSyncTensorsGraph(coll, std::move(parameters_data),
                                  std::move(tensors_data),
                                  std::move(cached_computation));
}

void XLATensor::SyncTensorsGraph(std::vector<XLATensor>* tensors,
                                 absl::Span<const std::string> devices,
                                 bool wait, bool sync_xla_data) {
  TF_VLOG(4) << "Trying to sync the value of " << tensors->size()
             << " tensor(s)";
  tensorflow::profiler::TraceMe activity(
      "SyncTensorsGraph", tensorflow::profiler::TraceMeLevel::kInfo);
  static const bool op_by_op =
      xla::sys_util::GetEnvBool("XLA_SYNC_TENSORS_OPBYOP", false);
  SyncTensorsConfig config;
  config.sync_xla_data = sync_xla_data;
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

void XLATensor::SyncLiveTensorsGraph(const torch::lazy::BackendDevice* device,
                                     absl::Span<const std::string> devices,
                                     bool wait) {
  tensorflow::profiler::TraceMe activity(
      "SyncLiveTensorsGraph", tensorflow::profiler::TraceMeLevel::kInfo);
  auto tensors = GetLiveTensors(device);
  TF_VLOG(4) << tensors.size() << " live tensors: devices=("
             << absl::StrJoin(devices, ",") << ")";
  SyncTensorsGraph(&tensors, devices, wait, /*sync_xla_data=*/true);
}

void XLATensor::MarkStep(const torch::lazy::BackendDevice& device) {
  XLA_COUNTER("MarkStep", 1);
  DeviceContextArena::Get()->MarkStep(device);
  ir::ScopePusher::ResetScopes();
  g_tls_data.Reset();
}

void XLATensor::WaitDeviceOps(absl::Span<const std::string> devices) {
  std::set<torch::lazy::BackendDevice> wait_devices;
  if (!devices.empty()) {
    for (auto& device_str : devices) {
      wait_devices.insert(torch::lazy::BackendDevice(device_str));
    }
  } else {
    for (auto& device_str : xla::ComputationClient::Get()->GetLocalDevices()) {
      wait_devices.insert(torch::lazy::BackendDevice(device_str));
    }
  }
  // The LockDevices() API returns a vector of xla::util::ExceptionCleanup
  // object, which is going to be freed immediately, turning this operation into
  // a lock barrier.
  LockDevices(wait_devices);
}

XLATensor::OpByOpAsync XLATensor::SyncTensorsGraphOpByOp(
    std::vector<XLATensor>* tensors, absl::Span<const std::string> devices,
    const SyncTensorsConfig& config) {
  struct Async {
    explicit Async(SyncTensorCollection coll,
                   std::vector<xla::ComputationClient::DataPtr> tensors_data,
                   std::vector<ir::Value> roots,
                   absl::Span<const std::string> devices)
        : coll(std::move(coll)),
          tensors_data(std::move(tensors_data)),
          roots(std::move(roots)),
          devices(devices.begin(), devices.end()) {}

    SyncTensorCollection coll;
    std::vector<xla::ComputationClient::DataPtr> tensors_data;
    std::vector<ir::Value> roots;
    std::vector<std::string> devices;
  };

  SyncTensorCollection coll = CollectSyncTensors(*tensors, config);
  DebugUtil::SaveTensorsGraphInfo("SyncTensorsGraphOpByOp", *tensors,
                                  &coll.indices);

  std::vector<ir::Value> roots = CollectRoots(*tensors, coll.indices);
  auto tensors_data = FetchTensorData(tensors, coll.config, coll.indices);
  auto async = std::make_shared<Async>(std::move(coll), std::move(tensors_data),
                                       std::move(roots), devices);

  auto syncfn = [async]() -> int {
    try {
      TF_VLOG(3) << "Executing (OpByOp) IR graph hash "
                 << torch::lazy::HashToString(async->coll.hash) << " on device "
                 << async->coll.device << " ...";
      std::vector<xla::ComputationClient::DataPtr> results =
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
      std::exception_ptr exptr = std::current_exception();
      for (auto& unlocker : async->coll.unlocker) {
        unlocker.SetStatus(exptr);
      }
      throw;
    }
    return 0;
  };
  OpByOpAsync async_op(std::move(syncfn));
  return async_op.Schedule();
}

void XLATensor::BuildInputOutputAliases(const std::vector<XLATensor>& tensors,
                                        absl::Span<const size_t> indices,
                                        ir::LoweringContext* lowering_ctx) {
  std::unordered_map<int64_t, size_t> output_tensor_id_map;
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t tensor_index = indices[i];
    int64_t tensor_id = tensors[tensor_index].GetUniqueId();
    output_tensor_id_map[tensor_id] = i;
  }
  const std::vector<xla::ComputationClient::DataPtr>& parameters_data =
      lowering_ctx->GetParametersData();
  std::vector<ssize_t> alias_map(indices.size(), -1);
  for (size_t i = 0; i < parameters_data.size(); ++i) {
    DeviceDataInfo* data_info =
        dynamic_cast<DeviceDataInfo*>(parameters_data[i]->info());
    if (data_info != nullptr && !data_info->read_only) {
      auto it = output_tensor_id_map.find(data_info->tensor_id);
      if (it != output_tensor_id_map.end()) {
        size_t output_index = it->second;
        xla::XlaOp root = lowering_ctx->GetResult(output_index);
        const xla::Shape& root_shape = XlaHelpers::ShapeOfXlaOp(root);
        if (parameters_data[i]->shape() == root_shape &&
            alias_map[output_index] < 0) {
          lowering_ctx->builder()->SetUpAlias(
              {static_cast<int64_t>(output_index)}, i, {});
          alias_map[output_index] = i;

          TF_VLOG(6) << "Aliased paramter " << i << " with output "
                     << output_index << ": " << parameters_data[i]->shape();
        }
      }
    }
  }
  XLA_VALUE_METRIC("InputOutputAliasCount", alias_map.size());
}

XLATensor::CompilationResult XLATensor::Compile(
    const std::vector<XLATensor>& tensors,
    absl::Span<const std::string> devices, const SyncTensorCollection& coll,
    PostOrderData* po_data) {
  tensorflow::profiler::TraceMe activity(
      [&] {
        return tensorflow::profiler::TraceMeEncode(
            "XLATensor::Compile",
            {{"graph_hash", torch::lazy::HashToString(coll.hash)}});
      },
      tensorflow::profiler::TraceMeLevel::kInfo);
  static const bool enable_aliasing =
      xla::sys_util::GetEnvBool("XLA_ENABLE_PARAM_ALIASING", true);
  ir::LoweringContext lowering_ctx("SyncTensorsGraph", coll.device,
                                   po_data->post_order,
                                   std::move(po_data->emission_map));
  for (auto index : coll.indices) {
    ir::Value ir_value = tensors[index].CurrentIrValue();
    xla::XlaOp root = lowering_ctx.GetOutputOp(
        torch::lazy::Output(ir_value.node.get(), ir_value.index));
    lowering_ctx.AddResult(root);
  }
  if (enable_aliasing && coll.config.sync_xla_data) {
    // We can only alias at the step barrier, when force_xla_data is true.
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
    // But, when we issue a step barrier (force_xla_data == true) we have to
    // turn everything into DEVICE_DATA, so we can activate aliasing.
    BuildInputOutputAliases(tensors, coll.indices, &lowering_ctx);
  }

  xla::XlaComputation computation = ConsumeValue(lowering_ctx.Build());
  xla::ProgramShape program_shape = ConsumeValue(computation.GetProgramShape());
  xla::Shape shape = MakeShapeWithDeviceLayout(program_shape.result(),
                                               static_cast<XlaDeviceType>(coll.device.type()));

  std::vector<xla::ComputationClient::CompileInstance> instances;
  instances.push_back({std::move(computation), coll.device.toString(),
                       xla::ComputationClient::Get()->GetCompilationDevices(
                           coll.device.toString(), devices),
                       &shape});

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
  XLA_CHECK_EQ(program_shape.parameters_size(),
               po_data->parameters_data.size());

  return {/*device=*/coll.device,
          /*emitted_nodes=*/lowering_ctx.GetEmittedNodeCount(),
          /*computation=*/std::move(computations.front()),
          /*parameters_data=*/std::move(po_data->parameters_data)};
}

std::shared_ptr<XLATensor::Async> XLATensor::SyncTensorsGraphInternal(
    std::vector<XLATensor>* tensors, absl::Span<const std::string> devices,
    const SyncTensorsConfig& config) {
  tensorflow::profiler::TraceMe activity(
      "SyncTensorsGraphInternal", tensorflow::profiler::TraceMeLevel::kInfo);
  SyncTensorCollection coll = CollectSyncTensors(*tensors, config);
  if (coll.indices.empty()) {
    return nullptr;
  }
  DebugUtil::SaveTensorsGraphInfo("ScheduleSyncTensorsGraph", *tensors,
                                  &coll.indices);

  PostOrderData po_data = RunPostOrder(*tensors, coll.indices);
  coll.hash = torch::lazy::HashCombine(
      coll.hash, torch::lazy::Hash(po_data.parameter_sequence));
  TF_VLOG(4) << "Parameter sequence graph hash "
             << torch::lazy::HashToString(coll.hash);
  std::shared_ptr<Async> async = TryRunCachedSync(tensors, &coll, &po_data);
  if (async != nullptr) {
    return async;
  }

  CompilationResult compile_result = Compile(*tensors, devices, coll, &po_data);

  XLA_VALUE_METRIC("TensorsGraphSize", compile_result.emitted_nodes);
  TF_VLOG(5) << "TensorsGraphSize=" << compile_result.emitted_nodes;

  auto cached_computation = std::make_shared<CachedComputation>(
      std::move(compile_result.computation));
  GetComputationCache()->Add(coll.hash, cached_computation);

  return ScheduleSyncTensorsGraph(
      tensors, &coll, std::move(compile_result.parameters_data),
      compile_result.device.toString(), std::move(cached_computation));
}

int64_t XLATensor::GetNextTensorId() {
  static std::atomic<int64_t>* id_generator = new std::atomic<int64_t>(1);
  return id_generator->fetch_add(1);
}

ir::Value XLATensor::GetRngSeed(const torch::lazy::BackendDevice& device) {
  return DeviceContextArena::Get()->GetRngSeed(device);
}

void XLATensor::SetRngSeed(const torch::lazy::BackendDevice& device, uint64_t seed) {
  DeviceContextArena::Get()->SetRngSeed(device, seed);
}

uint64_t XLATensor::GetRunningSeed(const torch::lazy::BackendDevice& device) {
  return DeviceContextArena::Get()->GetRunningSeed(device);
}

bool XLATensor::UseEagerDebugMode() {
  static const bool use_eager_debug_mode =
      xla::sys_util::GetEnvBool("XLA_USE_EAGER_DEBUG_MODE", false);
  return use_eager_debug_mode;
}

bool XLATensor::ShouldSyncIrNode() {
  if (!this->data()->ir_value) {
    return false;
  }
  return this->data()->ir_value->op() != ir::ops::xla_device_data;
}

}  // namespace torch_xla
