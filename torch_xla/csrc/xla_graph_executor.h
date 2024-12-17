#ifndef XLA_TORCH_XLA_CSRC_XLA_GRAPH_EXECUTOR_H_
#define XLA_TORCH_XLA_CSRC_XLA_GRAPH_EXECUTOR_H_

#include <c10/core/SymNodeImpl.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/lazy/core/ir_util.h>

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "absl/synchronization/blocking_counter.h"
#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/debug_util.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/runtime/cache.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/util.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/view.h"
#include "xla/client/xla_builder.h"
#include "xla/types.h"

namespace torch_xla {

class XLAGraphExecutor : public torch::lazy::LazyGraphExecutor {
 public:
  static XLAGraphExecutor* Get();

  // Override to use our own DeviceContextArena.
  void RegisterTensor(
      std::shared_ptr<torch::lazy::LazyTensor::Data> data) final;
  void UnregisterTensor(torch::lazy::LazyTensor::Data* data) final;

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
  void ApplyEagerSync(std::vector<XLATensorPtr>& tensors);

  // We don't use the upstream GetDeviceDataIrValue to have the
  // xla::PrimitiveType.
  torch::lazy::Value GetDeviceDataIrValue(
      const at::Scalar& value, xla::PrimitiveType type,
      const torch::lazy::BackendDevice& device);
  torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, xla::PrimitiveType type,
      const torch::lazy::BackendDevice& device);
  torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, const torch::lazy::BackendDevice& device);
  torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, xla::PrimitiveType type,
      absl::Span<const int64_t> dimensions,
      const torch::lazy::BackendDevice& device);
  torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, xla::PrimitiveType type,
      c10::SymIntArrayRef sym_size, const torch::lazy::BackendDevice& device);
  torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, const xla::Shape& shape,
      const torch::lazy::BackendDevice& device);
  torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, const xla::Shape& shape,
      std::optional<at::ScalarType> logical_element_type,
      const torch::lazy::BackendDevice& device);
  torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, const xla::Shape& shape,
      SymIntElements size_elements,
      std::optional<at::ScalarType> logical_element_type,
      const torch::lazy::BackendDevice& device);

  // Override to use our own DeviceContextArena.
  torch::lazy::Value GetRngSeed(const torch::lazy::BackendDevice& device) final;
  void SetRngSeed(const torch::lazy::BackendDevice& device,
                  uint64_t seed) final;
  uint64_t GetRunningSeed(const torch::lazy::BackendDevice& device) final;
  torch::lazy::BackendDataPtr GetBaseSeedData(
      const torch::lazy::BackendDevice& device);

  void SetAliasWithBufferDonorConfig(bool should_alias);

  bool GetAliasWithBufferDonorConfig();

  // Dumps the XLA HLO text of the computation accumulated in the graph which is
  // attached the tensors.
  // We don't use upstream DumpBackendComputation given we have our own format.
  std::string DumpHloComputation(const std::vector<XLATensorPtr>& tensors,
                                 EmitMode mode = EmitMode::kHloReadable);

  // Retrieves the set of XLA tensors which are currently live in the system,
  // for the given device. If device is nullptr, the live tensors for all
  // devices will be returned. Returned tensors are sorted by device as primary
  // key, and by unique ID as secondary key.
  // Unlike the base class, here we return XLATensorPtrs.
  std::vector<XLATensorPtr> GetLiveTensors(
      const torch::lazy::BackendDevice* device);

  // Applies all the pending IR operations queued over the input tensors. All
  // the tensors must be on the same device. If wait is true, the sync operation
  // will be run synchronously. The devices argument, if not empty, tells the
  // devices which should be participating into the replicated computation.
  void SyncTensorsGraph(std::vector<XLATensorPtr>* tensors,
                        absl::Span<const std::string> devices, bool wait,
                        bool sync_ltc_data, bool warm_up_cache_only = false);

  // Makes sure that any outstanding IR operation accumulated over live tensors,
  // gets turned into device data. If wait is true, the sync operation will be
  // run synchronously. The devices argument, if not empty, tells the devices
  // which should be participating into the replicated computation.
  // Override to enable profiling.
  void SyncLiveTensorsGraph(const torch::lazy::BackendDevice* device,
                            c10::ArrayRef<std::string> devices,
                            bool wait) final;

  // Marks an execution step, which allows the tensor framework to understand
  // the computation boundaries.
  // Override to use our own DeviceContextArena.
  void MarkStep(const torch::lazy::BackendDevice& device, bool reset_scope);

  // Waits for all the outstanding operations on all the supplied devices.
  // If devices is empty, the wait will happen for all local devices.
  // We don't use the WaitDeviceOps given we use local devices instead of
  // active devices.
  void WaitDeviceOps(absl::Span<const std::string> devices);

  // Retrieves the PyTorch CPU tensors behind the XLA tensors IR operations.
  // All the tensors must be on the same device.
  std::vector<at::Tensor> GetTensors(std::vector<XLATensorPtr>* tensors);

  // We don't use the upstream GetGraphHash as XLATensorPtr is used instead.
  torch::lazy::hash_t GetGraphHash(const std::vector<XLATensorPtr>& tensors);

  void MaybeDumpGraph(std::string name, torch::lazy::hash_t hash);

  // We don't use the upstream CachedComputation type given all fields are
  // different.
  struct CachedComputation {
    CachedComputation(runtime::ComputationClient::ComputationPtr computation,
                      bool is_sharded = false)
        : computation(std::move(computation)), is_sharded(is_sharded) {}

    runtime::ComputationClient::ComputationPtr computation;
    bool is_sharded;
  };

  using ComputationCache =
      runtime::util::AbstractCache<torch::lazy::hash_t, CachedComputation,
                                   torch::lazy::HashReducer>;
  using MemoryCache =
      runtime::util::Cache<torch::lazy::hash_t, CachedComputation,
                           torch::lazy::HashReducer>;
  using PersistentCache =
      runtime::util::PersistentCache<torch::lazy::hash_t, CachedComputation,
                                     torch::lazy::HashReducer>;

  ComputationCache* GetComputationCache();
  bool IsComputationCacheInitialized();

  std::vector<torch::lazy::BackendDataPtr> ExecuteComputationWithBarrier(
      torch::lazy::hash_t hash, const std::vector<at::IValue>& graph_inputs,
      const torch::lazy::BackendDevice& device);

  std::vector<torch::lazy::BackendDataPtr> ExecuteStablehlo(
      std::string stablehlo_bytecode,
      const std::vector<at::IValue>& graph_inputs,
      const torch::lazy::BackendDevice& device);

  void ClearPendingIrs(std::vector<XLATensorPtr> tensors,
                       const torch::lazy::BackendDevice& device);

  void SetUseEagerMode(bool use_eager_mode) {
    use_eager_mode_ = use_eager_mode;
  }

  bool UseEagerMode() { return use_eager_mode_; }

  void SetAllowExecution(bool allow_execution) {
    allow_execution_ = allow_execution;
  }

  bool AllowExecution() { return allow_execution_; }

  void SetCurrentGraphName(std::string graph_name) {
    current_graph_name_ = graph_name;
  }

  std::string CurrentGraphName() { return current_graph_name_; }

 private:
  // This is just to group results from compile(). Since our computation is
  // different, we don't reuse the upstream CompilationResult.
  struct CompilationResult {
    torch::lazy::BackendDevice device;
    size_t emitted_nodes = 0;
    runtime::ComputationClient::ComputationPtr computation;
    std::vector<torch::lazy::BackendDataPtr> parameters_data;
    bool is_sharded = false;
  };

  struct Async : public torch::lazy::LazyGraphExecutor::Async {
    Async(SyncTensorCollection* coll,
          std::vector<torch::lazy::BackendDataPtr> parameters_data,
          std::vector<torch::lazy::BackendDataPtr> tensors_data,
          ComputationCache::TypePtr cached_computation);

    ComputationCache::TypePtr cached_computation;
  };

  class DeviceContextArena
      : public torch::lazy::LazyGraphExecutor::DeviceContextArena {
   public:
    static DeviceContextArena* Get();

    // This method returns XLATensorPtrs instead of LazyTensorPtrs.
    std::vector<XLATensorPtr> GetLiveTensors(
        const torch::lazy::BackendDevice* device);

    // We override this to use our own + and * for torch::lazy::Value.
    torch::lazy::Value GetRngSeed(
        const torch::lazy::BackendDevice& device) final;

    torch::lazy::BackendDataPtr GetBaseSeedData(
        const torch::lazy::BackendDevice& device);

    void SetAliasWithBufferDonorConfig(bool should_alias) {
      should_alias_with_buffer_donor = should_alias;
    }

    bool GetAliasWithBufferDonorConfig() {
      return should_alias_with_buffer_donor;
    }

    void SaveGraphAsString(
        torch::lazy::hash_t hash, absl::Span<const XLATensorPtr> tensors,
        const std::vector<size_t>* indices,
        DebugUtil::GraphFormat format = DebugUtil::GetDefaultGraphFormat());

    void SaveOutputShapes(torch::lazy::hash_t hash,
                          std::vector<xla::Shape> outptu_shapes);

    std::string GetGraphByHash(torch::lazy::hash_t hash);

    // Return shapes is a pointer to the saved vector. Caller should be careful
    // if this pointer will be saved and access later since the value might be
    // changed. This should be fine in most cases since PyTorch/XLA tracing is
    // signle threaded.
    std::vector<xla::Shape>* GetOutputShapesByHash(torch::lazy::hash_t hash);

   private:
    // Below two maps are used for dynamo integration.
    std::unordered_map<torch::lazy::hash_t, std::string,
                       torch::lazy::HashReducer>
        hash_to_graph_map;
    std::unordered_map<torch::lazy::hash_t, std::vector<xla::Shape>,
                       torch::lazy::HashReducer>
        hash_to_output_shape_map;
    // We override this to use TensorToXlaData().
    torch::lazy::Value IrValueFromScalar(
        const at::Scalar& value, at::ScalarType scalar_type,
        const torch::lazy::BackendDevice& device) final;
    bool should_alias_with_buffer_donor = false;
  };

  XLAGraphExecutor() = default;

  // We don't use upstream CollectSyncTensors as we need to enable GSPMD.
  SyncTensorCollection CollectSyncTensors(
      const std::vector<XLATensorPtr>& tensors,
      const SyncTensorsConfig& config);

  // Waits for this SyncTensorCollection's device barrier and acquire the lock.
  // Override to enable SPMD.
  void TensorCollectionBarrier(SyncTensorCollection* coll) final;

  // Gathers the XLA device data for all the input tensors, after an
  // asynchronous operation.
  // TODO(alanwaketan): Reuse the upstream one once Functionalization is done.
  std::vector<torch::lazy::BackendDataPtr> GatherTensorsXlaData(
      const std::vector<XLATensorPtr>& tensors,
      absl::Span<const size_t> indices,
      absl::Span<const torch::lazy::BackendDataPtr> tensors_data);

  // TODO(alanwaketan): Reuse the upstream one once Functionalization is done.
  std::vector<torch::lazy::Value> CollectRoots(
      const std::vector<XLATensorPtr>& tensors,
      absl::Span<const size_t> indices);

  // TODO(alanwaketan): Reuse the upstream one once Functionalization is done.
  std::vector<torch::lazy::BackendDataPtr> SetTensorData(
      std::vector<XLATensorPtr>* tensors, const SyncTensorsConfig& config,
      absl::Span<const size_t> indices,
      const std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec);

  // We don't use upstream ExtractIRAndPrepareTensorData as we need to
  // instantiate xla::shape.
  void ExtractIRAndPrepareXlaData_(
      std::vector<XLATensorPtr>* tensors, const SyncTensorsConfig& config,
      const absl::Span<const size_t> indices,
      std::vector<torch::lazy::Value>& ir_values,
      std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec);

  // We don't use upstream FetchTensors as we have xla::Literal.
  std::vector<at::Tensor> FetchTensors(std::vector<XLATensorPtr>* tensors,
                                       absl::Span<const xla::Literal> literals,
                                       const std::vector<size_t>* indices);

  // Schedules the execution of a sync tensors operation in background. The
  // asynchronous operation will hold the device locks by capturing the ones
  // present within the coll structure.
  // We don't use the upstream ScheduleSyncTensorsGraph since
  // our CachedComputation is different from upstream.
  std::shared_ptr<Async> ScheduleSyncTensorsGraph(
      SyncTensorCollection* coll,
      std::vector<torch::lazy::BackendDataPtr> parameters_data,
      std::vector<torch::lazy::BackendDataPtr> tensors_data,
      std::vector<XLATensor::ShardingSpecPtr> sharding_specs,
      ComputationCache::TypePtr cached_computation);
  std::shared_ptr<Async> ScheduleSyncTensorsGraph(
      std::vector<XLATensorPtr>* tensors, SyncTensorCollection* coll,
      std::vector<torch::lazy::BackendDataPtr> parameters_data,
      std::string device, ComputationCache::TypePtr cached_computation,
      const std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec);

  // Override to enable profiler.
  PostOrderData RunPostOrder(const std::vector<torch::lazy::Value>& ir_values,
                             SyncTensorCollection* coll) final;

  // We don't use the upstream LookupCachedCompile since
  // our CachedComputation is different from upstream.
  ComputationCache::TypePtr LookupCachedCompile(
      const torch::lazy::hash_t& hash);

  // We don't use the upstream TryRunCachedSync since
  // our CachedComputation is different from upstream.
  std::pair<bool, std::shared_ptr<Async>> TryRunCachedSync(
      std::vector<XLATensorPtr>* tensors, SyncTensorCollection* coll,
      PostOrderData* po_data,
      const std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec,
      bool warm_up_cache_only);

  std::vector<size_t> GetBufferDonors(
      const std::vector<XLATensorPtr>& tensors,
      const SyncTensorCollection& coll,
      const std::vector<torch::lazy::BackendDataPtr>& parameters_data);

  void SetBufferDonors(LoweringContext* lowering_ctx,
                       const std::vector<size_t>& buffer_donor_indices);

  // TODO(yeounoh) auto-sharding can change tensors shardings, which needs to be
  // accounted for in Dynamo integration.
  CompilationResult Compile(std::vector<XLATensorPtr>& tensors,
                            absl::Span<const std::string> devices,
                            const SyncTensorCollection& coll,
                            PostOrderData* po_data,
                            const std::vector<torch::lazy::Value>& ir_values,
                            const std::vector<size_t>& buffer_donor_indices);

  // We don't use the upstream SyncTensorsGraphInternal since
  // our CachedComputation is different from upstream.
  std::shared_ptr<Async> SyncTensorsGraphInternal(
      std::vector<XLATensorPtr>* tensors, absl::Span<const std::string> devices,
      const SyncTensorsConfig& config, bool warm_up_cache_only = false);

  ComputationCache* computation_cache_;
  bool use_eager_mode_ = false;
  bool allow_execution_ = true;
  std::string current_graph_name_ = "";
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_XLA_GRAPH_EXECUTOR_H_
