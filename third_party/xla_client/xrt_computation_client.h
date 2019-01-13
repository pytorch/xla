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
#include "tensorflow/contrib/tpu/proto/topology.pb.h"
#include "tensorflow/core/framework/tensor.h"

namespace xla {

class XrtComputationClient : public ComputationClient {
  struct DeviceHandle {
    DeviceHandle(string device, int64 handle)
        : device(std::move(device)), handle(handle) {}

    string device;
    int64 handle;
  };

  struct XrtHandle {
    XrtHandle(XrtComputationClient* self, int64 handle)
        : self(self), handle(handle), released(false) {}

    absl::optional<int64> Release() {
      if (released.exchange(true)) {
        return absl::nullopt;
      }
      return handle;
    }

    XrtComputationClient* self;
    int64 handle;
    std::atomic<bool> released;
  };

  struct XrtData : public Data, public XrtHandle {
    XrtData(XrtComputationClient* self, string device, Shape device_shape,
            int64 handle)
        : Data(std::move(device), std::move(device_shape)),
          XrtHandle(self, handle) {}

    ~XrtData() override {
      if (!released) {
        self->ReleaseXrtData(this);
      }
    }
  };

  struct XrtComputation : public Computation, public XrtHandle {
    XrtComputation(XrtComputationClient* self, XlaComputation computation,
                   ProgramShape program_shape, std::vector<string> devices,
                   int64 handle, string compilation_device)
        : Computation(std::move(computation), std::move(program_shape),
                      std::move(devices)),
          XrtHandle(self, handle),
          compilation_device(std::move(compilation_device)) {}

    ~XrtComputation() override {
      if (!released) {
        self->ReleaseXrtComputation(this);
      }
    }

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

    string name;
    int task_no;
  };

  struct Options {
    string default_device;
    // Maps a PyTorch device ID (example, "GPU:0", "TPU:0") to the full
    // coordinates in TF device format
    // (ie, /job:tpu_worker/replica:0/task:0/device:TPU:0), of the worker
    // exposing that device.
    std::map<string, string> device_map;
    // Maps a TPU Worker with an HOST:PORT string.
    std::map<Worker, string> workers_map;
  };

  XrtComputationClient(Options options);

  std::vector<std::shared_ptr<Data>> TransferToServer(
      tensorflow::gtl::ArraySlice<const LiteralDevice> literals) override;

  std::vector<Literal> TransferFromServer(
      tensorflow::gtl::ArraySlice<const std::shared_ptr<Data>> handles)
      override;

  std::vector<std::shared_ptr<Computation>> Compile(
      std::vector<CompileInstance> instances) override;

  std::vector<std::shared_ptr<Data>> ExecuteComputation(
      const Computation& computation,
      tensorflow::gtl::ArraySlice<Data*> arguments, const string& device,
      const ExecuteComputationOptions& options) override;

  std::vector<std::vector<std::shared_ptr<Data>>> ExecuteReplicated(
      const Computation& computation,
      const std::vector<std::vector<Data*>>& arguments,
      tensorflow::gtl::ArraySlice<const string> devices,
      const ExecuteReplicatedOptions& options) override;

  std::vector<std::vector<std::shared_ptr<Data>>> ExecuteParallel(
      tensorflow::gtl::ArraySlice<const Computation* const> computations,
      const std::vector<std::vector<Data*>>& arguments,
      tensorflow::gtl::ArraySlice<const string> devices,
      const ExecuteParallelOptions& options) override;

  std::vector<std::vector<std::shared_ptr<Data>>> DeconstructTuple(
      tensorflow::gtl::ArraySlice<const std::shared_ptr<Data>> tuples) override;

  string GetDefaultDevice() const override;

 private:
  // When we split a batch operation into per-session batches, we use this data
  // structure to collect the per-session work.
  struct SessionWork {
    tensorflow::ClientSession::FeedType feed_inputs;
    std::vector<tensorflow::Output> outputs_handles;
    std::vector<tensorflow::Operation> operations;
    std::vector<size_t> index_mapping;
  };

  XrtSession* GetSessionForTarget(const string& target,
                                  XrtSessionCache::SessionMap* session_map);
  XrtSession* GetSessionForXrtDevice(const string& xrt_device,
                                     XrtSessionCache::SessionMap* session_map);
  XrtSession* GetSessionForDevice(const string& device,
                                  XrtSessionCache::SessionMap* session_map);

  string GetEffectiveDevice(const string& device) const;

  const string& TorchDeviceToXrtDevice(const string& device) const;

  string GetCompilationDevice(
      tensorflow::gtl::ArraySlice<const string> devices) const;

  std::unique_ptr<xrt::XLAComputation> CreateXrtComputation(
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<const string> devices,
      const Shape* output_shape) const;

  tensorflow::Tensor GetArgumentsInputs(
      tensorflow::gtl::ArraySlice<Data*> arguments, const string& device,
      tensorflow::ClientSession::FeedType* feed_inputs);

  std::vector<tensorflow::Output> CreateExecuteOps(
      XrtSessionCache::SessionMap* session_map,
      tensorflow::gtl::ArraySlice<const Computation* const> computations,
      const std::vector<std::vector<Data*>>& arguments, bool explode_tuple,
      tensorflow::gtl::ArraySlice<const string> devices,
      tensorflow::ClientSession::FeedType* feed_inputs);

  std::vector<tensorflow::Output> CreateExecuteOps(
      XrtSessionCache::SessionMap* session_map,
      const XrtComputation& computation,
      const std::vector<std::vector<Data*>>& arguments, bool explode_tuple,
      tensorflow::gtl::ArraySlice<const string> devices,
      tensorflow::ClientSession::FeedType* feed_inputs);

  std::vector<std::vector<std::shared_ptr<Data>>> RunComputations(
      const XrtSessionCache::SessionMap& session_map,
      const std::vector<tensorflow::Output>& exec_ops,
      tensorflow::gtl::ArraySlice<const Computation* const> computations,
      tensorflow::gtl::ArraySlice<const string> devices,
      const tensorflow::ClientSession::FeedType& feed_inputs);

  // Retrieves the worker,worker_host pair for a given PyTorch device (ie,
  // TPU:0).
  std::pair<Worker, string> GetWorkerForDevice(const string& xrt_device) const;

  // Retrieves the worker,worker_host pair for a given XRT device (ie,
  // /job:tpu_worker/replica:0/task:0/device:TPU:0).
  std::pair<Worker, string> GetWorkerForXrtDevice(
      const string& xrt_device) const;

  void ReleaseHandles(
      std::vector<DeviceHandle>* handles,
      const std::function<const XrtSession::CachedNode&(
          XrtSession*, const tensorflow::Scope&, const string&)>& op_generator,
      metrics::Metric* timed_metric, metrics::Counter* destroy_counter);

  bool ReleaseHandle(XrtHandle* handle, const string& device,
                     std::vector<DeviceHandle>* handles);

  bool ReleaseXrtData(XrtData* xrt_data);

  bool ReleaseXrtComputation(XrtComputation* xrt_computation);

  // Starts the handle releaser thread (which runs the HandleReleaser() API).
  void StartHandleReleaser();

  // The handler releaser function. Runs in the releaser thread and never
  // returns.
  void HandleReleaser();

  // Retrieves the mesh coordinates of a given XRT device.
  const std::vector<int>& GetDeviceMeshCoords(const string& xrt_device) const;

  tensorflow::tpu::TopologyProto InitializeAndFetchTopology(
      const string& xrt_device);

  void InitializeDevices();

  std::vector<std::shared_ptr<Data>> GetComputationResults(
      const tensorflow::Tensor& xrt_result, const Shape& result_shape,
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
  //  holders[0] = XLA Computation handle place-holder (DT_INT^%)
  //  holders[1] = xrt::XRTExecutionConfig place-holder (DT_STRING)
  //  holders[2] = Inputs for the XRTExecute (DT_INT64[])
  const XrtSession::CachedNode& GetExecuteNode(XrtSession* session,
                                               const tensorflow::Scope& scope,
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

  // Converts an XLA data type to a tensorflow data type.
  static tensorflow::DataType XlaTypeToDataType(PrimitiveType dtype);

  static tensorflow::TensorShape MakeEquivalentTensorShape(const Shape& shape);

  // Builds an argument vector usable in a replicated context, out of a single
  // replica argument vector. Essentially turns a [N] into a [1][N].
  static std::vector<std::vector<Data*>> BuildParallelArguments(
      tensorflow::gtl::ArraySlice<Data*> arguments);

  // Extracts the XlaComputation pointers out of Computation ones. Used to be
  // passed to xrt_util::CheckComputationStatus() for its error reporting.
  static std::vector<const XlaComputation*> GetXlaComputations(
      tensorflow::gtl::ArraySlice<const Computation* const> computations);

  // Checks whether a local GRPC service is required, and starts it if need it.
  static void MaybeCreateLocalService(
      const XrtComputationClient::Options& options);

  Options options_;
  std::mutex lock_;
  std::map<string, std::vector<int>> device_mesh_coords_;
  XrtSessionCache session_cache_;
  std::unique_ptr<xla_util::TriggeredTask> triggered_task_;
  util::Cache<string, std::shared_ptr<Computation>,
              util::PartialHasher<string, 4096>>
      compilation_cache_;
  // Access to the following members must be done while holding lock_.
  // XRT thread safety semantics.
  std::vector<DeviceHandle> released_data_handles_;
  std::vector<DeviceHandle> released_compile_handles_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_XRT_COMPUTATION_CLIENT_H_
