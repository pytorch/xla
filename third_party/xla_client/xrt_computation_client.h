#ifndef XLA_CLIENT_XRT_COMPUTATION_CLIENT_H_
#define XLA_CLIENT_XRT_COMPUTATION_CLIENT_H_

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "absl/types/optional.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/xla/xla_client/cache.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/mesh_service.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/triggered_task.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/xla/xla_client/xrt_local_service.h"
#include "tensorflow/compiler/xla/xla_client/xrt_session.h"
#include "tensorflow/compiler/xla/xla_client/xrt_session_cache.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xrt/cc/ops/xrt_compile_ops.h"
#include "tensorflow/compiler/xrt/cc/ops/xrt_execute_op.h"
#include "tensorflow/compiler/xrt/cc/ops/xrt_state_ops.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/tpu/topology.pb.h"

namespace xla {

class XrtLocker {
 public:
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

  std::mutex mutex_;
  std::condition_variable cv_;
  bool locked_ = false;
  std::exception_ptr exptr_;
};

class DataHandleLocker : public XrtLocker {
 public:
  static const int64_t dummy_handle;
};

class XrtComputationClient : public ComputationClient {
  struct DeviceHandle {
    std::string device;
    int64_t handle;
  };

  class XrtHandle {
   public:
    XrtHandle(int64_t handle, std::function<void(int64_t)> releaser,
              bool async = false)
        : handle_(handle), releaser(std::move(releaser)) {
      if (async) {
        locker = std::make_shared<DataHandleLocker>();
      } else {
        locker = nullptr;
      }
    }

    ~XrtHandle() {
      // Handle might only contain dummy value, need to wait for the
      // true handle assigniment.
      if (locker) {
        XLA_TIMED("HandleBarrierWait");
        locker->Barrier();
      }
      releaser(handle_);
    }

    // Lock the current XrtHandle and prevent other caller from accessing the
    // handle_ value. This function will return an ExceptionCleanup object which
    // will rethrow the exception if there is one and unlock the XrtHandle upon
    // destruction.
    xla::util::ExceptionCleanup LockHandle() {
      std::shared_ptr<DataHandleLocker> locker_copy = this->locker;
      locker_copy->Lock();
      return xla::util::ExceptionCleanup(
          [locker_copy = std::move(locker_copy)](
              xla::util::ExceptionCleanup::StatusType status) {
            locker_copy->Unlock(std::move(status));
          });
    }

    void update_handle(int64_t handle) {
      // handle can only be updated once when it is dummy.
      XLA_CHECK_EQ(handle_, DataHandleLocker::dummy_handle);
      handle_ = handle;
    }

    int64_t handle() {
      // Handle might only contain dummy value, need to wait for the
      // true handle assigniment
      if (locker) {
        XLA_TIMED("HandleBarrierWait");
        locker->Barrier();
      }
      return handle_;
      ;
    }

   private:
    int64_t handle_;
    std::shared_ptr<DataHandleLocker> locker;
    std::function<void(int64_t)> releaser;
  };

  using XrtHandlePtr = std::shared_ptr<XrtHandle>;

  struct XrtData : public Data {
    XrtData(std::string device, Shape device_shape)
        : Data(std::move(device), std::move(device_shape)),
          handle_ptr(nullptr) {}
    XrtData(XrtComputationClient* self, std::string device, Shape device_shape,
            int64_t handle)
        : Data(std::move(device), std::move(device_shape)),
          handle_ptr(std::make_shared<XrtHandle>(
              handle, [self, device = this->device()](int64_t handle) {
                self->ReleaseXrtData(device, handle);
              })) {}

    XrtData(XrtComputationClient* self, std::string device, Shape device_shape,
            XrtHandlePtr handle)
        : Data(std::move(device), std::move(device_shape)),
          handle_ptr(handle) {}

    int64_t get_handle() const {
      XLA_CHECK(HasValue());
      return handle_ptr->handle();
    }

    OpaqueHandle GetOpaqueHandle() override { return get_handle(); }

    void Assign(const Data& data) override;

    bool HasValue() const override { return handle_ptr != nullptr; }

    XrtHandlePtr handle_ptr;
  };

  struct XrtComputation : public Computation {
    XrtComputation(XrtComputationClient* self, XlaComputation computation,
                   ProgramShape program_shape, std::vector<std::string> devices,
                   int64_t handle, std::string compilation_device)
        : Computation(std::move(computation), std::move(program_shape),
                      std::move(devices)),
          handle_ptr(std::make_shared<XrtHandle>(
              handle, [self, compilation_device = std::move(
                                 compilation_device)](int64_t handle) {
                self->ReleaseXrtComputation(compilation_device, handle);
              })) {}

    int64_t get_handle() const { return handle_ptr->handle(); }

    XrtHandlePtr handle_ptr;
  };

 public:
  struct Device {
    Device() = default;
    Device(const std::string& device_str);

    std::string kind;
    int ordinal = 0;
  };

  struct Worker {
    Worker(std::string name, int task_no)
        : name(std::move(name)), task_no(task_no) {}

    bool operator<(const Worker& rhs) const {
      if (task_no != rhs.task_no) {
        return task_no < rhs.task_no;
      }
      return name.compare(rhs.name) < 0;
    }

    bool operator==(const Worker& rhs) const {
      return task_no == rhs.task_no && name == rhs.name;
    }

    std::string name;
    int task_no;
  };

  struct Options {
    std::string default_device;
    // Maps a PyTorch device ID (example, "GPU:0", "TPU:0") to the full
    // coordinates in TF device format
    // (ie, /job:tpu_worker/replica:0/task:0/device:TPU:0), of the worker
    // exposing that device. These devices are all the devices present within
    // the TPU mesh.
    std::map<std::string, std::string> global_device_map;
    // These are the devices that this instance of PyTorch is handling. These
    // devices are in the form of "CPU:0", "TPU:3", ... For each of these
    // devices, there is an entry within the global_device_map.
    std::set<std::string> devices;
    // Maps a TPU Worker with an EndPoint.
    std::map<Worker, std::string> workers_map;
  };

  XrtComputationClient(
      Options options,
      std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto);

  DataPtr CreateDataPlaceholder(std::string device, Shape shape) override;

  std::vector<DataPtr> CreateAsyncDatas(
      absl::Span<const TensorSource> tensors) override;

  std::vector<xla::util::ExceptionCleanup> LockAsyncDatas(
      absl::Span<const DataPtr> datas) override;

  std::vector<DataPtr> GetDataShards(DataPtr data) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  std::vector<DataPtr> TransferToServer(
      absl::Span<const TensorSource> tensors) override;

  void TransferToServer(absl::Span<const TensorSource> tensors,
                        absl::Span<const DataPtr> datas) override;

  DataPtr TransferShardsToServer(absl::Span<const TensorSource> tensor_shards,
                                 std::string device,
                                 xla::Shape shape) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  DataPtr CopyToDevice(DataPtr data, std::string dst) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  std::vector<Literal> TransferFromServer(
      absl::Span<const DataPtr> handles) override;

  std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) override;

  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      const std::string& device,
      const ExecuteComputationOptions& options) override;

  std::vector<std::vector<DataPtr>> ExecuteReplicated(
      const Computation& computation,
      const std::vector<std::vector<DataPtr>>& arguments,
      absl::Span<const std::string> devices,
      const ExecuteReplicatedOptions& options) override;

  std::vector<std::vector<DataPtr>> ExecuteParallel(
      absl::Span<const Computation* const> computations,
      const std::vector<std::vector<DataPtr>>& arguments,
      absl::Span<const std::string> devices,
      const ExecuteParallelOptions& options) override;

  std::vector<DataPtr> ExecuteChained(absl::Span<const ExecuteChainedOp> ops,
                                      const std::string& device) override;

  std::vector<std::vector<DataPtr>> DeconstructTuple(
      absl::Span<const DataPtr> tuples) override;

  std::string GetResourceDomain(const std::string& device) const override;

  std::string GetDefaultDevice() const override;

  size_t GetNumDevices() const override;

  std::vector<std::string> GetLocalDevices() const override;

  std::vector<std::string> GetAllDevices() const override;

  int GetProcessIndex() const override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  int GetNumProcesses() const override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  const absl::flat_hash_map<std::string,
                            xla::ComputationClient::DeviceAttribute>&
  GetDeviceAttributes(const std::string& device) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  void SetReplicationDevices(
      std::shared_ptr<std::vector<std::string>> devices) override;

  std::shared_ptr<std::vector<std::string>> GetReplicationDevices() override;

  void SetRngSeed(size_t seed) override;

  std::map<std::string, Metric> GetMetrics() const override;

  MemoryInfo GetMemoryInfo(const std::string& device) override;

  void PrepareToExit() override;

  static Worker ParseWorker(const std::string& worker);

  static std::string GetMultiProcessingDevice();

 private:
  // The data structure used for the key in the compilation cache. Compilations
  // handles are valid within given domain (essentially the host+port worker
  // endpoints), so the key must include the domain.
  struct CompilationCacheKey {
    struct Hash {
      size_t operator()(const CompilationCacheKey& entry) const {
        util::PartialHasher<std::string, 4096> hasher;
        hash_t h = util::DataHash(entry.domain.data(), entry.domain.size());
        return util::HashReduce(
            util::HashCombine(h, hasher(entry.serialized_computation)));
      }
    };

    CompilationCacheKey(std::string domain, std::string serialized_computation)
        : domain(std::move(domain)),
          serialized_computation(std::move(serialized_computation)) {}
    CompilationCacheKey() = default;
    CompilationCacheKey(CompilationCacheKey&&) = default;
    CompilationCacheKey& operator=(CompilationCacheKey&&) = default;
    bool operator==(const CompilationCacheKey& rhs) const {
      return domain == rhs.domain &&
             serialized_computation == rhs.serialized_computation;
    }

    std::string domain;
    std::string serialized_computation;
  };

  // When we split a batch operation into per-session batches, we use this data
  // structure to collect the per-session work.
  struct SessionWork {
    tensorflow::ClientSession::FeedType feed_inputs;
    std::vector<tensorflow::Output> outputs_handles;
    std::vector<tensorflow::Operation> operations;
    std::vector<size_t> index_mapping;
  };

  XrtSession* GetSessionForTarget(XrtSessionCache* cache,
                                  const std::string& target,
                                  XrtSessionCache::SessionMap* session_map);
  XrtSession* GetSessionForXrtDevice(XrtSessionCache* cache,
                                     const std::string& xrt_device,
                                     XrtSessionCache::SessionMap* session_map);
  XrtSession* GetSessionForDevice(XrtSessionCache* cache,
                                  const std::string& device,
                                  XrtSessionCache::SessionMap* session_map);

  const std::string& TorchDeviceToXrtDevice(const std::string& device) const;

  template <typename T>
  void SetupExecConfig(const Device& device, T* exec_config) const;

  std::unique_ptr<xrt::XLAComputation> CreateXrtComputation(
      const XlaComputation& computation, absl::Span<const std::string> devices,
      const Shape* output_shape) const;

  tensorflow::Tensor GetArgumentsInputs(absl::Span<const DataPtr> arguments,
                                        const std::string& device);

  std::vector<tensorflow::Output> CreateExecuteOps(
      XrtSessionCache::SessionMap* session_map,
      absl::Span<const Computation* const> computations,
      const std::vector<std::vector<DataPtr>>& arguments, bool explode_tuple,
      absl::Span<const std::string> devices,
      tensorflow::ClientSession::FeedType* feed_inputs);

  std::vector<tensorflow::Output> CreateExecuteOps(
      XrtSessionCache::SessionMap* session_map,
      const XrtComputation& computation,
      const std::vector<std::vector<DataPtr>>& arguments, bool explode_tuple,
      absl::Span<const std::string> devices,
      tensorflow::ClientSession::FeedType* feed_inputs);

  std::vector<std::vector<DataPtr>> RunComputations(
      const XrtSessionCache::SessionMap& session_map,
      const std::vector<tensorflow::Output>& exec_ops,
      absl::Span<const Computation* const> computations,
      absl::Span<const std::string> devices,
      const tensorflow::ClientSession::FeedType& feed_inputs);

  std::vector<DataPtr> TransferToServerHelper(
      absl::Span<const TensorSource> tensors, absl::Span<const DataPtr> datas);

  std::vector<DataPtr> TransferToServerInternal(
      absl::Span<const TensorSource> tensors, absl::Span<const DataPtr> datas);

  // Retrieves the worker,worker_host pair for a given PyTorch device (ie,
  // TPU:0).
  std::pair<Worker, std::string> GetWorkerForDevice(
      const std::string& device) const;

  // Retrieves the worker,worker_host pair for a given XRT device (ie,
  // /job:tpu_worker/replica:0/task:0/device:TPU:0).
  std::pair<Worker, std::string> GetWorkerForXrtDevice(
      const std::string& xrt_device) const;

  void ReleaseHandles(std::vector<DeviceHandle>* handles,
                      const std::function<const XrtSession::CachedNode&(
                          XrtSession*, const tensorflow::Scope&,
                          const std::string&)>& op_generator,
                      metrics::Metric* timed_metric,
                      metrics::Counter* destroy_counter);

  void ReleaseHandle(int64_t handle, const std::string& device,
                     std::vector<DeviceHandle>* handles);

  void ReleaseXrtData(const std::string& device, int64_t handle);

  void ReleaseXrtComputation(const std::string& compilation_device,
                             int64_t handle);

  // Starts the handle releaser thread (which runs the HandleReleaser() API).
  void StartHandleReleaser();

  // The handler releaser function. Runs in the releaser thread and never
  // returns.
  void HandleReleaser();

  // Retrieves the mesh coordinates of a given XRT device.
  const std::vector<int>& GetDeviceMeshCoords(
      const std::string& xrt_device) const;

  void InitializeDevices(
      std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto);

  service::grpc::Config CreateMeshServiceConfig(
      const tensorflow::tpu::TopologyProto* topology_proto) const;

  void CreateMeshService(const std::string& address,
                         const tensorflow::tpu::TopologyProto* topology_proto);

  void SetupGpuRuntime();

  std::vector<DataPtr> GetComputationResults(
      const tensorflow::Tensor& xrt_result, const Shape& result_shape,
      const std::string& device);

  void InitSession(XrtSession* session) const;

  // Implement the chained execution using the XRTExecuteChained op support.
  std::vector<DataPtr> ExecuteChainedXrt(absl::Span<const ExecuteChainedOp> ops,
                                         const std::string& device);

  // Implement the chained execution using multiple XRTExecute in many RPC round
  // trips.
  std::vector<DataPtr> ExecuteChainedSplit(
      absl::Span<const ExecuteChainedOp> ops, const std::string& device);

  // Creates an XRT graph with an XRTCompile operation:
  //
  //  XRTCompile(
  //    holders[0]
  //  )
  //
  // With:
  //  holders[0] = XLA Computation place-holder (DT_STRING)
  const XrtSession::CachedNode& GetCompileNode(XrtSession* session,
                                               const tensorflow::Scope& scope,
                                               const std::string& device) const;

  // Creates an XRT graph with an XRTExecute operation:
  //
  //  XRTExecute(
  //    holders[0],
  //    holders[1],
  //    holders[2]
  //  )
  //
  // With:
  //  holders[0] = XLA Computation handle place-holder (DT_INT64)
  //  holders[1] = xrt::XRTExecutionConfig place-holder (DT_STRING)
  //  holders[2] = Inputs for the XRTExecute (DT_INT64[])
  const XrtSession::CachedNode& GetExecuteNode(XrtSession* session,
                                               const tensorflow::Scope& scope,
                                               const std::string& device) const;

  // Creates an XRT graph with an XRTExecute operation:
  //
  //  XRTExecuteChained(
  //    holders[0],
  //    holders[1]
  //  )
  //
  // With:
  //  holders[0] = xrt::XRTChainedExecutePlan place-holder (DT_STRING)
  //  holders[1] = xrt::XRTChainedExecuteConfig place-holder (DT_STRING)
  const XrtSession::CachedNode& GetExecuteChainedNode(
      XrtSession* session, const tensorflow::Scope& scope,
      const std::string& device) const;

  // Creates an XRT graph with an XRTReadLiteral operation:
  //
  //  XRTReadLiteral(
  //    holders[0]
  //  )
  //
  // With:
  //  holders[0] = The handle place-holder to be read (DT_INT64)
  const XrtSession::CachedNode& GetReadNode(XrtSession* session,
                                            const tensorflow::Scope& scope,
                                            const std::string& device) const;

  // Creates an XRTAllocateFromTensor node for creating a device tensor with
  // the given shape and layout:
  //
  //  XRTAllocateFromTensor(
  //    holders[0]
  //  )
  //
  // With:
  //  holders[0] = Tensor place-holder (DT_* - depends on shape type)
  const XrtSession::CachedNode& GetAllocateNode(XrtSession* session,
                                                const tensorflow::Scope& scope,
                                                const std::string& device,
                                                const Shape& shape) const;

  // Creates an XRTReleaseAllocationHandle node:
  //
  //  XRTReleaseAllocationHandle(
  //    holders[0]
  //  )
  //
  // With:
  //  holders[0] = To be released handle place-holder (DT_INT64)
  const XrtSession::CachedNode& GetReleaseAllocationHandleNode(
      XrtSession* session, const tensorflow::Scope& scope,
      const std::string& device) const;

  // Creates an XRTReleaseCompilationHandle node:
  //
  //  XRTReleaseCompilationHandle(
  //    holders[0]
  //  )
  //
  // With:
  //  holders[0] = To be released compilation handle place-holder (DT_INT64)
  const XrtSession::CachedNode& GetReleaseCompileHandleNode(
      XrtSession* session, const tensorflow::Scope& scope,
      const std::string& device) const;

  // Creates an XRTSubTuple node:
  //
  //  XRTSubTuple(
  //    holders[0],
  //    holders[1]
  //  )
  //
  // With:
  //  holders[0] = Tuple handle place-holder (DT_INT64)
  //  holders[1] = Tuple index place-holder (DT_INT32[])
  const XrtSession::CachedNode& GetSubTupleNode(
      XrtSession* session, const tensorflow::Scope& scope,
      const std::string& device) const;

  // Creates an XRTMemoryInfo node:
  //
  //  XRTMemoryInfo()
  const XrtSession::CachedNode& GetMemoryInfoNode(
      XrtSession* session, const tensorflow::Scope& scope,
      const std::string& device) const;

  // Checks the result of a compile operation, and dumps the XLA computation
  // graphs in case of error.
  static void CheckCompileStatus(const Status& status,
                                 const std::vector<CompileInstance>& instances,
                                 const SessionWork& session_work);

  // Converts an XLA data type to a tensorflow data type.
  static tensorflow::DataType XlaTypeToDataType(PrimitiveType dtype);

  static tensorflow::TensorShape MakeEquivalentTensorShape(const Shape& shape);

  // Builds an argument vector usable in a replicated context, out of a single
  // replica argument vector. Essentially turns a [N] into a [1][N].
  static std::vector<std::vector<DataPtr>> BuildParallelArguments(
      absl::Span<const DataPtr> arguments);

  static std::vector<size_t> PartitionTransferToServer(
      absl::Span<const TensorSource> tensors);

  // Extracts the XlaComputation pointers out of Computation ones. Used to be
  // passed to xrt_util::CheckComputationStatus() for its error reporting.
  static std::vector<const XlaComputation*> GetXlaComputations(
      absl::Span<const Computation* const> computations);

  static tensorflow::ConfigProto CreateConfigProto(const Options& options);

  static tensorflow::tpu::TopologyProto InitializeAndFetchTopology(
      const std::string& job, int task_no, const std::string& worker_host_port,
      const tensorflow::ConfigProto& config);

  static std::string GetLocalTarget(const Options& options);

  // Checks whether a local GRPC service is required, and starts it if need it.
  void MaybeCreateLocalService(const Options& options);

  Options options_;
  std::mutex lock_;
  std::map<std::string, std::vector<int>> device_mesh_coords_;
  std::unique_ptr<XrtSessionCache> session_cache_;
  std::unique_ptr<XrtSessionCache> alloc_session_cache_;
  std::unique_ptr<util::TriggeredTask> triggered_task_;
  XrtLocalService* local_service_ = nullptr;
  util::Cache<CompilationCacheKey, Computation, CompilationCacheKey::Hash>
      compilation_cache_;
  std::atomic<size_t> rng_seed_;
  // Access to the following members must be done while holding lock_.
  // XRT thread safety semantics.
  std::vector<DeviceHandle> released_data_handles_;
  std::vector<DeviceHandle> released_compile_handles_;
  // The mesh service which is used to coordinate all the client hosts which are
  // feeding different TPU devices in a POD (or slice) training.
  std::unique_ptr<service::MeshService> mesh_service_;
  std::shared_ptr<std::vector<std::string>> replication_devices_;
};

}  // namespace xla

#endif  // XLA_CLIENT_XRT_COMPUTATION_CLIENT_H_
