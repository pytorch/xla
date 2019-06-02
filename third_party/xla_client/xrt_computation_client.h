#ifndef TENSORFLOW_COMPILER_XLA_RPC_XRT_COMPUTATION_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_RPC_XRT_COMPUTATION_CLIENT_H_

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

class XrtComputationClient : public ComputationClient {
  struct DeviceHandle {
    string device;
    int64 handle;
  };

  struct XrtHandle {
    XrtHandle(XrtComputationClient* self, int64 handle)
        : self(self), handle(handle) {}

    XrtComputationClient* self;
    int64 handle;
  };

  using XrtHandlePtr = std::shared_ptr<XrtHandle>;

  struct XrtData : public Data {
    XrtData(XrtComputationClient* self, string device, Shape device_shape)
        : Data(std::move(device), std::move(device_shape)) {}
    XrtData(XrtComputationClient* self, string device, Shape device_shape,
            int64 handle)
        : Data(std::move(device), std::move(device_shape)),
          handle_ptr(std::make_shared<XrtHandle>(self, handle)) {}

    ~XrtData() override {
      if (handle_ptr != nullptr && handle_ptr.use_count() == 1) {
        handle_ptr->self->ReleaseXrtData(this);
      }
    }

    int64 get_handle() const { return handle_ptr->handle; }

    void Assign(const Data& data) override;

    bool HasValue() const override { return handle_ptr != nullptr; }

    XrtHandlePtr handle_ptr;
  };

  struct XrtComputation : public Computation {
    XrtComputation(XrtComputationClient* self, XlaComputation computation,
                   ProgramShape program_shape, std::vector<string> devices,
                   int64 handle, string compilation_device)
        : Computation(std::move(computation), std::move(program_shape),
                      std::move(devices)),
          handle_ptr(std::make_shared<XrtHandle>(self, handle)),
          compilation_device(std::move(compilation_device)) {}

    ~XrtComputation() override {
      if (handle_ptr.use_count() == 1) {
        handle_ptr->self->ReleaseXrtComputation(this);
      }
    }

    int64 get_handle() const { return handle_ptr->handle; }

    XrtHandlePtr handle_ptr;
    string compilation_device;
  };

 public:
  struct Worker {
    Worker(string name, int task_no)
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

    string name;
    int task_no;
  };

  struct Options {
    string default_device;
    // Maps a PyTorch device ID (example, "GPU:0", "TPU:0") to the full
    // coordinates in TF device format
    // (ie, /job:tpu_worker/replica:0/task:0/device:TPU:0), of the worker
    // exposing that device. These devices are all the devices present within
    // the TPU mesh.
    std::map<string, string> global_device_map;
    // These are the devices that this instance of PyTorch is handling. These
    // devices are in the form of "CPU:0", "TPU:3", ... For each of these
    // devices, there is an entry within the global_device_map.
    std::set<string> devices;
    // Maps a TPU Worker with an EndPoint.
    std::map<Worker, string> workers_map;
  };

  XrtComputationClient(
      Options options,
      std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto);

  DataPtr CreateDataPlaceholder(string device, Shape shape) override;

  std::vector<DataPtr> TransferToServer(
      tensorflow::gtl::ArraySlice<const TensorSource> tensors) override;

  std::vector<Literal> TransferFromServer(
      tensorflow::gtl::ArraySlice<const DataPtr> handles) override;

  std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) override;

  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation,
      tensorflow::gtl::ArraySlice<const DataPtr> arguments,
      const string& device, const ExecuteComputationOptions& options) override;

  std::vector<std::vector<DataPtr>> ExecuteReplicated(
      const Computation& computation,
      const std::vector<std::vector<DataPtr>>& arguments,
      tensorflow::gtl::ArraySlice<const string> devices,
      const ExecuteReplicatedOptions& options) override;

  std::vector<std::vector<DataPtr>> ExecuteParallel(
      tensorflow::gtl::ArraySlice<const Computation* const> computations,
      const std::vector<std::vector<DataPtr>>& arguments,
      tensorflow::gtl::ArraySlice<const string> devices,
      const ExecuteParallelOptions& options) override;

  std::vector<DataPtr> ExecuteChained(
      tensorflow::gtl::ArraySlice<const ExecuteChainedOp> ops,
      const string& device) override;

  std::vector<std::vector<DataPtr>> DeconstructTuple(
      tensorflow::gtl::ArraySlice<const DataPtr> tuples) override;

  string GetResourceDomain(const string& device) const override;

  string GetDefaultDevice() const override;

  size_t GetNumDevices() const override;

  std::vector<string> GetLocalDevices() const override;

  std::vector<string> GetAllDevices() const override;

  void SetRngSeed(size_t seed) override;

 private:
  // The data structure used for the key in the compilation cache. Compilations
  // handles are valid within given domain (essentially the host+port worker
  // endpoints), so the key must include the domain.
  struct CompilationCacheKey {
    struct Hash {
      size_t operator()(const CompilationCacheKey& entry) const {
        util::PartialHasher<string, 4096> hasher;
        return tensorflow::Hash64(entry.domain.data(), entry.domain.size(),
                                  hasher(entry.serialized_computation));
      }
    };

    CompilationCacheKey(string domain, string serialized_computation)
        : domain(std::move(domain)),
          serialized_computation(std::move(serialized_computation)) {}
    CompilationCacheKey() = default;
    CompilationCacheKey(CompilationCacheKey&&) = default;
    CompilationCacheKey& operator=(CompilationCacheKey&&) = default;
    bool operator==(const CompilationCacheKey& rhs) const {
      return domain == rhs.domain &&
             serialized_computation == rhs.serialized_computation;
    }

    string domain;
    string serialized_computation;
  };

  // When we split a batch operation into per-session batches, we use this data
  // structure to collect the per-session work.
  struct SessionWork {
    tensorflow::ClientSession::FeedType feed_inputs;
    std::vector<tensorflow::Output> outputs_handles;
    std::vector<tensorflow::Operation> operations;
    std::vector<size_t> index_mapping;
  };

  XrtSession* GetSessionForTarget(XrtSessionCache* cache, const string& target,
                                  XrtSessionCache::SessionMap* session_map);
  XrtSession* GetSessionForXrtDevice(XrtSessionCache* cache,
                                     const string& xrt_device,
                                     XrtSessionCache::SessionMap* session_map);
  XrtSession* GetSessionForDevice(XrtSessionCache* cache, const string& device,
                                  XrtSessionCache::SessionMap* session_map);

  string GetEffectiveDevice(const string& device) const;

  const string& TorchDeviceToXrtDevice(const string& device) const;

  std::unique_ptr<xrt::XLAComputation> CreateXrtComputation(
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<const string> devices,
      const Shape* output_shape) const;

  tensorflow::Tensor GetArgumentsInputs(
      tensorflow::gtl::ArraySlice<const DataPtr> arguments,
      const string& device);

  std::vector<tensorflow::Output> CreateExecuteOps(
      XrtSessionCache::SessionMap* session_map,
      tensorflow::gtl::ArraySlice<const Computation* const> computations,
      const std::vector<std::vector<DataPtr>>& arguments, bool explode_tuple,
      tensorflow::gtl::ArraySlice<const string> devices,
      tensorflow::ClientSession::FeedType* feed_inputs);

  std::vector<tensorflow::Output> CreateExecuteOps(
      XrtSessionCache::SessionMap* session_map,
      const XrtComputation& computation,
      const std::vector<std::vector<DataPtr>>& arguments, bool explode_tuple,
      tensorflow::gtl::ArraySlice<const string> devices,
      tensorflow::ClientSession::FeedType* feed_inputs);

  std::vector<std::vector<DataPtr>> RunComputations(
      const XrtSessionCache::SessionMap& session_map,
      const std::vector<tensorflow::Output>& exec_ops,
      tensorflow::gtl::ArraySlice<const Computation* const> computations,
      tensorflow::gtl::ArraySlice<const string> devices,
      const tensorflow::ClientSession::FeedType& feed_inputs);

  // Retrieves the worker,worker_host pair for a given PyTorch device (ie,
  // TPU:0).
  std::pair<Worker, string> GetWorkerForDevice(const string& device) const;

  // Retrieves the worker,worker_host pair for a given XRT device (ie,
  // /job:tpu_worker/replica:0/task:0/device:TPU:0).
  std::pair<Worker, string> GetWorkerForXrtDevice(
      const string& xrt_device) const;

  void ReleaseHandles(
      std::vector<DeviceHandle>* handles,
      const std::function<const XrtSession::CachedNode&(
          XrtSession*, const tensorflow::Scope&, const string&)>& op_generator,
      metrics::Metric* timed_metric, metrics::Counter* destroy_counter);

  void ReleaseHandle(int64 handle, const string& device,
                     std::vector<DeviceHandle>* handles);

  void ReleaseXrtData(XrtData* xrt_data);

  void ReleaseXrtComputation(XrtComputation* xrt_computation);

  // Starts the handle releaser thread (which runs the HandleReleaser() API).
  void StartHandleReleaser();

  // The handler releaser function. Runs in the releaser thread and never
  // returns.
  void HandleReleaser();

  // Retrieves the mesh coordinates of a given XRT device.
  const std::vector<int>& GetDeviceMeshCoords(const string& xrt_device) const;

  tensorflow::tpu::TopologyProto InitializeAndFetchTopology(
      const string& job, int task_no, const string& worker_host_port);

  void InitializeDevices(
      std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto);

  void CreateMeshService(const tensorflow::tpu::TopologyProto& topology_proto);

  std::vector<DataPtr> GetComputationResults(
      const tensorflow::Tensor& xrt_result, const Shape& result_shape,
      const string& device);

  void InitSession(XrtSession* session) const;

  // Implement the chained execution using the XRTExecuteChained op support.
  std::vector<DataPtr> ExecuteChainedXrt(
      tensorflow::gtl::ArraySlice<const ExecuteChainedOp> ops,
      const string& device);

  // Implement the chained execution using multiple XRTExecute in many RPC round
  // trips.
  std::vector<DataPtr> ExecuteChainedSplit(
      tensorflow::gtl::ArraySlice<const ExecuteChainedOp> ops,
      const string& device);

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
                                               const string& device) const;

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
                                               const string& device) const;

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
      const string& device) const;

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
                                            const string& device) const;

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
                                                const string& device,
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
      const string& device) const;

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
      const string& device) const;

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
  const XrtSession::CachedNode& GetSubTupleNode(XrtSession* session,
                                                const tensorflow::Scope& scope,
                                                const string& device) const;

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
      tensorflow::gtl::ArraySlice<const DataPtr> arguments);

  // Extracts the XlaComputation pointers out of Computation ones. Used to be
  // passed to xrt_util::CheckComputationStatus() for its error reporting.
  static std::vector<const XlaComputation*> GetXlaComputations(
      tensorflow::gtl::ArraySlice<const Computation* const> computations);

  static tensorflow::ConfigProto CreateConfigProto(const Options& options);

  // Checks whether a local GRPC service is required, and starts it if need it.
  static void MaybeCreateLocalService(
      const XrtComputationClient::Options& options);

  Options options_;
  std::mutex lock_;
  std::map<string, std::vector<int>> device_mesh_coords_;
  std::unique_ptr<XrtSessionCache> session_cache_;
  std::unique_ptr<XrtSessionCache> alloc_session_cache_;
  std::unique_ptr<util::TriggeredTask> triggered_task_;
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
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_XRT_COMPUTATION_CLIENT_H_
