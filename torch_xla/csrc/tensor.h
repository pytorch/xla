#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "c10/core/SymNodeImpl.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/async_task.h"
#include "tensorflow/compiler/xla/xla_client/cache.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/lazy/core/ir_util.h"
#include "torch_xla/csrc/computation.h"
#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ir_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/view.h"

namespace torch_xla {

class TORCH_API XLASymNodeImpl : public c10::SymNodeImpl {
 public:
  XLASymNodeImpl(torch::lazy::NodePtr ptr) : node_(std::move(ptr)) {}
  bool is_int() override;
  bool is_float() override;
  c10::SymNode eq(const c10::SymNode& other) override;
  c10::SymNode add(const c10::SymNode& other) override;
  c10::SymNode mul(const c10::SymNode& other) override;
  c10::SymNode floordiv(const c10::SymNode& other) override;
  c10::SymNode wrap_int(int64_t num) override;

  torch::lazy::NodePtr node() { return node_; }
  std::string str() override;

  bool bool_() override;
  int64_t int_() override;

 private:
  torch::lazy::NodePtr node_;
};

class XLATensor;
using XLATensorPtr = c10::intrusive_ptr<XLATensor>;

class XLATensor : public torch::lazy::LazyTensor {
  class DeviceContextArena;

 public:
  // This is the core XLA tensor data structure where all the tensor data is
  // held. The XLA tensor is nothing more than a shared pointer to a Data
  // object.
  struct Data {
    Data(torch::lazy::BackendDataPtr xla_data,
         const torch::lazy::BackendDevice& device,
         c10::optional<at::ScalarType> logical_element_type)
        : xla_data(std::move(xla_data)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(torch::lazy::Value ir_value, const torch::lazy::BackendDevice& device,
         c10::optional<at::ScalarType> logical_element_type)
        : ir_value(std::move(ir_value)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(std::shared_ptr<View> view, const torch::lazy::BackendDevice& device,
         c10::optional<at::ScalarType> logical_element_type)
        : view(std::move(view)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(at::Tensor tensor_data, const torch::lazy::BackendDevice& device)
        : logical_element_type(tensor_data.scalar_type()),
          tensor_data(std::move(tensor_data)),
          device(device),
          unique_id(GetNextTensorId()) {}

    ~Data();

    torch::lazy::BackendDataPtr xla_data;
    torch::lazy::Value ir_value;
    std::shared_ptr<View> view;
    // TODO: remove this in favor of torch::lazy::Shape within ir_value.
    c10::optional<at::ScalarType> logical_element_type;
    c10::optional<at::Tensor> tensor_data;
    const torch::lazy::BackendDevice device;
    const int64_t unique_id = 0;
    size_t generation = 1;
  };

  static XLATensorPtr Create(const at::Tensor& tensor,
                             const torch::lazy::BackendDevice& device);
  static XLATensorPtr Create(
      torch::lazy::BackendDataPtr xla_data,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensorPtr Create(
      torch::lazy::Value ir_value, const torch::lazy::BackendDevice& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  // Create a new XLA tensor with the same metadata of the input tensor (with
  // possible overrides), and the new IR value.
  XLATensorPtr CreateFrom(torch::lazy::Value ir_value) const;
  XLATensorPtr CreateFrom(
      torch::lazy::Value ir_value,
      c10::optional<at::ScalarType> logical_element_type_opt) const;
  // TODO: We should remove this one once MaybeCastIrValue is no longer needed.
  XLATensorPtr CreateFrom(torch::lazy::Value ir_value,
                          const torch::lazy::BackendDevice& device,
                          at::ScalarType logical_element_type) const;

  // The default ctor previously created a null LazyTensor (one with no 'data'
  // obj). Creating a null XLATensor is no longer possible, since the same can
  // be achieved by creating a null LazyTensorPtr and it is way too confusing to
  // have to check both lazy_tensor_ptr && *lazy_tensor_ptr, so everywhere that
  // used to rely on a LazyTensor obj with a null Data can now rely on a null
  // LazyTensorPtr instead.
  XLATensor() = delete;

  size_t generation() const { return data()->generation; }

  XLATensorPtr alias() const {
    return c10::make_intrusive<XLATensor>(XLATensor(data_ptr()));
  }

  int64_t size(int64_t dim) const;

  at::Tensor ToTensor(bool detached);

  void ShallowCopyTo(XLATensorPtr dest) const;

  // Assigns the tensor value to the XLA tensor.
  void SetTensor(at::Tensor tensor);

  void UpdateFromTensor(at::Tensor tensor, bool sync);
  void UpdateFromTensorOut(at::Tensor tensor);
  void UpdateFromTensorOut(const XLATensorPtr& tensor);

  at::ScalarType dtype() const;
  c10::optional<at::ScalarType> dtype_optional() const;

  // Set logical_element_type which is visible to upstream PyTorch.
  void SetScalarType(c10::optional<at::ScalarType> logical_element_type);

  xla::util::MaybeRef<xla::Shape> shape() const;

  const torch::lazy::BackendDevice& GetDevice() const;
  int64_t GetUniqueId() const;

  // Retrieves an opaque ID of the alias object upon which the tensor's view is
  // rooted, or 0 if this tensor is not a view.
  std::ptrdiff_t GetViewAliasId() const;

  // Fetches the XLA data behind the tensor. If the tensor has a graph defining
  // its current value, executes the graph and fetches the XLA data result.
  torch::lazy::BackendDataPtr GetXlaData();

  // Fetches the current value of the XLA data, which can be missing (nullptr)
  // in case the tensor has a graph defining its current value,
  torch::lazy::BackendDataPtr CurrentXlaData() const;

  void SetXlaData(torch::lazy::BackendDataPtr xla_data);

  // Retrieves the current IR XlaNode, or nullptr in case no active IR XlaNode
  // is available.
  torch::lazy::Value CurrentIrValue() const;

  // Retrieves the IR Node representing this XLATensor. One will be created if
  // missing. Note that although this is a const API, it actually changes the
  // internal state of the object.
  torch::lazy::Value GetIrValue() const;

  void SetIrValue(torch::lazy::Value ir_value, bool inplace = true);
  void SetInPlaceIrValue(torch::lazy::Value ir_value);

  c10::optional<at::Tensor> CurrentTensorData() const;

  std::vector<XLATensorPtr> MakeOutputTensors(
      torch::lazy::NodePtr node, bool inherit_logical_type = true) const;

  void SetSubView(ViewInfo view_info) const;
  void ModifyCurrentView(ViewInfo view_info) const;
  XLATensorPtr CreateViewTensor(ViewInfo view_info) const;

  XLATensorPtr CopyTensorToDevice(const torch::lazy::BackendDevice& device);

  // Applies the queue of operations in preparation for using the data.
  void ApplyPendingGraph();

  Data* data() const;

  // This method just syncs the tensors passed as argument. This method is
  // called at two places:
  // 1. Creating tensor from IR value. This is where an output tensor is created
  // from an IR computation
  // 2. SetIRValue(). This is where the IR value of in place operations are
  // updated. Note: We do not sync the output of ViewTensors. This is because:
  // 1. The operations that generate the ViewTensor would be re-done when its
  // base tensor is updated. When the base tensor is updated, torch-xla would
  // apply all the views on it and hence the operations would be repeated.
  // Hence, we don't sync the ViewTensors and in case users want to print them,
  // they can still do it and will incur a small graph compile. This way we
  // avoid some extra compiles. This makes it lazy just for view operations.
  // Note: ViewTensors do not share the same storage as the input tensor. This
  // is by design. Currently, to respect the definitions of view tensors,
  // different view relationships between tensors is tracked and update all the
  // tensors to make it look as if they share same storage. Hence, the
  // operations on view tensor would be repeated when we try to sync the tensor
  // that is affected by the view tensor.
  static void ApplyEagerSync(std::vector<XLATensorPtr>& tensors);

  static torch::lazy::Value GetDeviceDataIrValue(
      const at::Scalar& value, xla::PrimitiveType type,
      const torch::lazy::BackendDevice& device);
  // Use with caution, constant will cause more frequent recompilation
  // compared to the device_data.
  static torch::lazy::Value GetIrValueForConstant(const at::Scalar& value,
                                                  const xla::Shape& shape);
  static torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, xla::PrimitiveType type,
      const torch::lazy::BackendDevice& device);
  static torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, const torch::lazy::BackendDevice& device);
  static torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, xla::PrimitiveType type,
      absl::Span<const int64_t> dimensions,
      const torch::lazy::BackendDevice& device);
  static torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, const xla::Shape& shape,
      const torch::lazy::BackendDevice& device);
  static torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, const xla::Shape& shape,
      c10::optional<at::ScalarType> logical_element_type,
      const torch::lazy::BackendDevice& device);

  static torch::lazy::Value GetRngSeed(
      const torch::lazy::BackendDevice& device);

  static void SetRngSeed(const torch::lazy::BackendDevice& device,
                         uint64_t seed);

  static uint64_t GetRunningSeed(const torch::lazy::BackendDevice& device);

  static torch::lazy::BackendDataPtr GetRngSeedData(
      const torch::lazy::BackendDevice& device, bool reset);

  // Dumps the XLA HLO text of the computation accumulated in the graph which is
  // attached the tensors.
  static std::string DumpHloComputation(
      const std::vector<XLATensorPtr>& tensors);

  // Retrieves the set of XLA tensors which are currently live in the system,
  // for the given device. If device is nullptr, the live tensors for all
  // devices will be returned. Returned tensors are sorted by device as primary
  // key, and by unique ID as secondary key.
  static std::vector<XLATensorPtr> GetLiveTensors(
      const torch::lazy::BackendDevice* device);

  // Applies all the pending IR operations queued over the input tensors. All
  // the tensors must be on the same device. If wait is true, the sync operation
  // will be run synchronously. The devices argument, if not empty, tells the
  // devices which should be participating into the replicated computation.
  static void SyncTensorsGraph(std::vector<XLATensorPtr>* tensors,
                               absl::Span<const std::string> devices, bool wait,
                               bool sync_xla_data);

  // Makes sure that any outstanding IR operation accumulated over live tensors,
  // gets turned into device data. If wait is true, the sync operation will be
  // run synchronously. The devices argument, if not empty, tells the devices
  // which should be participating into the replicated computation.
  static void SyncLiveTensorsGraph(const torch::lazy::BackendDevice* device,
                                   absl::Span<const std::string> devices,
                                   bool wait);

  // Marks an execution step, which allows the tensor framework to understand
  // the computation boundaries.
  static void MarkStep(const torch::lazy::BackendDevice& device);

  // Waits for all the outstanding operations on all the supplied devices.
  // If devices is empty, the wait will happen for all local devices.
  static void WaitDeviceOps(absl::Span<const std::string> devices);

  // Retrieves the PyTorch CPU tensors behind the XLA tensors IR operations.
  // All the tensors must be on the same device.
  static std::vector<at::Tensor> GetTensors(std::vector<XLATensorPtr>* tensors);

  // XLATensor sharding annotation. ShardingSpec wraps xla::OpSharding and
  // can be extended to hold other sharding information from the user.
  static torch::lazy::hash_t GetGraphHash(
      const std::vector<XLATensorPtr>& tensors);

  // XLA SPMD sharding spec annoation. The XLA tensor uses this to create
  // HloSharding for replication, manual and tile shardings.
  struct ShardingSpec {
    ShardingSpec(const xla::OpSharding& sharding) : sharding(sharding) {}

    xla::OpSharding sharding;
  };
  using ShardingSpecPtr = std::shared_ptr<ShardingSpec>;

  // Annotate the IR value with ShardingSpec.
  void SetShardingSpec(const ShardingSpec& sharding_spec);
  // Clear sharding annotation attached to the IR value and transfer sharded
  // data back to host.
  void ClearShardingSpec();
  ShardingSpecPtr sharding_spec() const;

  void SetStorage(const c10::Storage& storage) { storage_ = storage; }
  const c10::Storage& Storage() const { return storage_; }

  struct CachedComputation {
    CachedComputation(ComputationPtr computation, bool is_sharded = false)
        : computation(std::move(computation)), is_sharded(is_sharded) {}

    ComputationPtr computation;
    bool is_sharded;
  };

  using ComputationCache =
      xla::util::Cache<torch::lazy::hash_t, CachedComputation,
                       torch::lazy::HashReducer>;

  static ComputationCache* GetComputationCache();

  int64_t GetOpaqueHandle() const;

  static std::vector<torch::lazy::BackendDataPtr> ExecuteComputationWithBarrier(
      torch::lazy::ComputationPtr computation,
      c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
      const torch::lazy::BackendDevice& device);

  static void ClearPendingIrs(std::vector<XLATensorPtr> tensors,
                              const torch::lazy::BackendDevice& device);

 private:
  struct SyncTensorsConfig {
    // Whether we want to force XLA data on the target tensors (hence trimming
    // the IR graph above them).
    bool force_xla_data = true;
    // Whether when setting the XLA data, the other properties of the tensor
    // state should be reset.
    bool sync_xla_data = true;
  };

  struct SyncTensorCollection {
    SyncTensorCollection() : hash(0) {}

    SyncTensorsConfig config;
    std::vector<size_t> indices;
    torch::lazy::hash_t hash;
    std::vector<xla::util::ExceptionCleanup> unlocker;
    torch::lazy::BackendDevice device;
  };

  struct PostOrderData {
    std::vector<const torch::lazy::Node*> post_order;
    torch::lazy::Util::EmissionMap emission_map;
    std::vector<torch::lazy::BackendDataPtr> parameters_data;
    std::vector<size_t> parameter_sequence;
  };

  struct CompilationResult {
    torch::lazy::BackendDevice device;
    size_t emitted_nodes = 0;
    ComputationPtr computation;
    std::vector<torch::lazy::BackendDataPtr> parameters_data;
    bool is_sharded = false;
  };

  struct Async {
    Async(SyncTensorCollection* coll,
          std::vector<torch::lazy::BackendDataPtr> parameters_data,
          std::vector<torch::lazy::BackendDataPtr> tensors_data,
          ComputationCache::TypePtr cached_computation);

    void Wait();

    xla::util::MultiWait mwait;
    std::vector<size_t> indices;
    std::vector<xla::util::ExceptionCleanup> unlocker;
    std::vector<torch::lazy::BackendDataPtr> parameters_data;
    std::string device;
    ComputationCache::TypePtr cached_computation;
    std::vector<torch::lazy::BackendDataPtr> tensors_data;
  };

  XLATensor(const at::Tensor& tensor, const torch::lazy::BackendDevice& device);
  XLATensor(torch::lazy::BackendDataPtr xla_data,
            c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  XLATensor(torch::lazy::Value ir_value,
            const torch::lazy::BackendDevice& device,
            c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  XLATensor(std::shared_ptr<View> view,
            const torch::lazy::BackendDevice& device,
            c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  XLATensor(std::shared_ptr<Data> data);

  static XLATensorPtr Create(
      std::shared_ptr<View> view, const torch::lazy::BackendDevice& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  std::shared_ptr<Data> data_ptr() const { return data_; }

  void SetXlaData(torch::lazy::BackendDataPtr xla_data, bool sync);

  void AssignIrValue(torch::lazy::Value ir_value) const;

  void SetTensorData(at::Tensor tensor_data);

  torch::lazy::Value CreateTensorNode(torch::lazy::BackendDataPtr data,
                                      bool read_only) const;

  View::IrNode GetViewUpdate(const std::shared_ptr<View>& view) const;

  std::shared_ptr<View> UpdateView(std::shared_ptr<View> view,
                                   torch::lazy::Value ir_value) const;

  std::shared_ptr<View> CreateView(ViewInfo view_info) const;

  torch::lazy::Value MaybeCastIrValue(
      torch::lazy::Value ir_value, const torch::lazy::BackendDevice& device,
      c10::optional<at::ScalarType> logical_element_type) const;

  // We build an XLA graph accumulating XLA operations, but at a given point we
  // need to force a rendering, otherwise the graph can grow without control.
  // Think:
  //   for i in range(0, 100000):
  //     a = a + b
  void TryLimitGraphSize();

  torch::lazy::Value GetIrValueForTensor(
      const at::Tensor& tensor, const torch::lazy::BackendDevice& device) const;

  static SyncTensorCollection CollectSyncTensors(
      const std::vector<XLATensorPtr>& tensors,
      const SyncTensorsConfig& config);

  // Waits for this SyncTensorCollection's device barrier and acuire the lock.
  static void TensorCollectionBarrier(SyncTensorCollection* coll);

  // Implementation of the GetTensors() API using the op-by-op executor.
  static std::vector<at::Tensor> GetTensorsOpByOp(
      std::vector<XLATensorPtr>* tensors);

  static std::vector<at::Tensor> GetTensorsFused(
      std::vector<XLATensorPtr>* tensors);

  // Runs an asynchronous syn operation using the op-by-op executor.
  using OpByOpAsync = xla::util::AsyncTask<int>;
  static OpByOpAsync SyncTensorsGraphOpByOp(
      std::vector<XLATensorPtr>* tensors, absl::Span<const std::string> devices,
      const SyncTensorsConfig& config);

  // Gathers the XLA device data for all the input tensors, after an
  // asynchronous operation.
  static std::vector<torch::lazy::BackendDataPtr> GatherTensorsXlaData(
      const std::vector<XLATensorPtr>& tensors,
      absl::Span<const size_t> indices,
      absl::Span<const torch::lazy::BackendDataPtr> tensors_data);

  static std::vector<torch::lazy::Value> CollectRoots(
      const std::vector<XLATensorPtr>& tensors,
      absl::Span<const size_t> indices);

  static std::vector<torch::lazy::BackendDataPtr> SetTensorData(
      std::vector<XLATensorPtr>* tensors, const SyncTensorsConfig& config,
      absl::Span<const size_t> indices,
      const std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec);

  static void ExtractIRAndPrepareXlaData_(
      std::vector<XLATensorPtr>* tensors, const SyncTensorsConfig& config,
      const absl::Span<const size_t> indices,
      std::vector<torch::lazy::Value>& ir_values,
      std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec);

  static std::vector<at::Tensor> FetchTensors(
      std::vector<XLATensorPtr>* tensors,
      absl::Span<const xla::Literal> literals,
      const std::vector<size_t>* indices);

  // Schedules the execution of a sync tensors operation in background. The
  // asynchronous operation will hold the device locks by capturing the ones
  // present within the coll structure.
  static std::shared_ptr<XLATensor::Async> ScheduleSyncTensorsGraph(
      SyncTensorCollection* coll,
      std::vector<torch::lazy::BackendDataPtr> parameters_data,
      std::vector<torch::lazy::BackendDataPtr> tensors_data,
      ComputationCache::TypePtr cached_computation);

  static std::shared_ptr<Async> ScheduleSyncTensorsGraph(
      std::vector<XLATensorPtr>* tensors, SyncTensorCollection* coll,
      std::vector<torch::lazy::BackendDataPtr> parameters_data,
      std::string device, ComputationCache::TypePtr cached_computation,
      const std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec);

  static PostOrderData RunPostOrder(
      const std::vector<torch::lazy::Value>& ir_values,
      SyncTensorCollection* coll);

  static ComputationCache::TypePtr LookupCachedCompile(
      const std::vector<XLATensorPtr>& tensors,
      const torch::lazy::hash_t& hash);

  static std::shared_ptr<Async> TryRunCachedSync(
      std::vector<XLATensorPtr>* tensors, SyncTensorCollection* coll,
      PostOrderData* po_data,
      const std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec);

  static std::vector<std::pair<int64_t, int64_t>> BuildInputOutputAliases(
      const std::vector<XLATensorPtr>& tensors,
      absl::Span<const size_t> indices, LoweringContext* lowering_ctx);

  static CompilationResult Compile(
      const std::vector<XLATensorPtr>& tensors,
      absl::Span<const std::string> devices, const SyncTensorCollection& coll,
      PostOrderData* po_data, const std::vector<torch::lazy::Value>& ir_values);

  static std::shared_ptr<Async> SyncTensorsGraphInternal(
      std::vector<XLATensorPtr>* tensors, absl::Span<const std::string> devices,
      const SyncTensorsConfig& config);

  static int64_t GetNextTensorId();

  static bool UseEagerDebugMode();

  bool ShouldSyncIrNode();

  std::shared_ptr<Data> data_;
  // Temporarily used to suport Tensor.is_alias_of().
  // This is a fake storage that doesn't store anything.
  // Instead it serves as a marker to mark LazyTensors that
  // points to the same storage, and thus alias of each other.
  // FIXME(alanwaketan): Remove this once we have functionalization (bdhirsh).
  c10::Storage storage_;
};

}  // namespace torch_xla
