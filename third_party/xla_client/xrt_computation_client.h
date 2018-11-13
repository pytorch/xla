#ifndef TENSORFLOW_COMPILER_XLA_RPC_XRT_COMPUTATION_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_RPC_XRT_COMPUTATION_CLIENT_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/types/optional.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xrt/cc/ops/xrt_compile_ops.h"
#include "tensorflow/compiler/xrt/cc/ops/xrt_execute_op.h"
#include "tensorflow/compiler/xrt/cc/ops/xrt_state_ops.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/contrib/tpu/proto/topology.pb.h"

namespace xla {

class XrtComputationClient : public ComputationClient {
  struct DeviceHandle {
    DeviceHandle(string device, int64 handle)
        : device(std::move(device)), handle(handle) {}

    string device;
    int64 handle;
  };

  struct XrtData : public Data {
    using Releaser = std::function<void(XrtData*)>;

    XrtData(string device, int64 handle, Shape device_shape, Releaser releaser)
        : Data(std::move(device), std::move(device_shape)),
          handle(handle),
          releaser(std::move(releaser)) {}

    ~XrtData() override {
      if (releaser) {
        releaser(this);
      }
    }

    void Release() { releaser = nullptr; }

    int64 handle;
    Releaser releaser;
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

  std::shared_ptr<Data> ExecuteComputation(
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<Data*> arguments,
      const Shape* output_shape) override;

  std::vector<std::shared_ptr<Data>> ExecuteReplicated(
      const XlaComputation& computation,
      const std::vector<std::vector<Data*>>& arguments,
      const Shape* output_shape) override;

  std::vector<std::shared_ptr<Data>> ExecuteParallel(
      tensorflow::gtl::ArraySlice<const XlaComputation> computations,
      const std::vector<std::vector<Data*>>& arguments,
      tensorflow::gtl::ArraySlice<const Shape* const> output_shapes) override;

  std::vector<std::vector<std::shared_ptr<Data>>> DeconstructTuple(
      tensorflow::gtl::ArraySlice<const std::shared_ptr<Data>> tuples) override;

  string GetDefaultDevice() const override;

 private:
  struct SessionData {
    SessionData(const string& target)
        : root(tensorflow::Scope::NewRootScope()), session(root, target) {}

    tensorflow::Scope root;
    tensorflow::ClientSession session;
  };

  // A cached node captures that single node, or the mini-graph root node,
  // together with the place-holders necessary to feed the node/sub-graph.
  // The end-point node can be either a tensorflow Operation or an Output.
  struct CachedNode {
    CachedNode(tensorflow::Output output,
               std::vector<tensorflow::ops::Placeholder> holders)
        : output(std::move(output)), holders(std::move(holders)) {}
    CachedNode(tensorflow::Operation operation,
               std::vector<tensorflow::ops::Placeholder> holders)
        : operation(std::move(operation)), holders(std::move(holders)) {}

    absl::optional<tensorflow::Output> output;
    absl::optional<tensorflow::Operation> operation;
    std::vector<tensorflow::ops::Placeholder> holders;
  };

  // The node cache holds a set of CachedNode of the same kind (by the means of
  // the NodeTypes entries).
  struct NodeCache {
    bool empty() const { return next >= nodes.size(); }

    void add(std::unique_ptr<CachedNode> node) {
      nodes.push_back(std::move(node));
    }

    const CachedNode& get() {
      CHECK_LT(next, nodes.size());
      return *nodes[next++];
    }

    void rewind() { next = 0; }

    size_t next = 0;
    std::vector<std::unique_ptr<CachedNode>> nodes;
  };

  // Every "kind" of cached node (or group of nodes - mini graph), have an ID
  // entry here.
  enum class NodeTypes {
    kCompileExecute,
    kRead,
    kAllocate,
    kSubTuple,
    kReleaseAllocationHandle,
  };

  struct NodeCacheKey {
    NodeCacheKey(string device, NodeTypes type)
        : device(std::move(device)), type(type) {}

    bool operator<(const NodeCacheKey& rhs) const {
      return type != rhs.type ? (type < rhs.type)
                              : (device.compare(rhs.device) < 0);
    }

    string device;
    NodeTypes type;
  };

  // When we split a batch operation into per-session batches, we use this data
  // structure to collect the per-session work.
  struct SessionWork {
    std::vector<tensorflow::Output> outputs_handles;
    std::vector<size_t> index_mapping;
  };

  struct ExecuteContext {
    ExecuteContext(tensorflow::Output execute_output, Shape result_shape)
        : execute_output(std::move(execute_output)),
          result_shape(std::move(result_shape)) {}

    tensorflow::Output execute_output;
    Shape result_shape;
  };

  SessionData* GetSessionForTarget(const string& target);
  SessionData* GetSessionForXrtDevice(const string& xrt_device);
  SessionData* GetSessionForDevice(const string& device);

  string GetEffectiveDevice(const string& device) const;

  const string& TorchDeviceToXrtDevice(const string& device) const;

  std::unique_ptr<xrt::XLAComputation> CreateXrtComputation(
      const XlaComputation& computation, int64 num_replicas,
      const std::vector<string>& devices, const Shape* output_shape) const;

  // Retrieves the unique, common, device for all the inputs. Issue a CHECK if
  // the inputs are not on a common device, as we cannot create an XLA
  // computation spanning multiple devices ATM.
  string GetArgumentsDevice(tensorflow::gtl::ArraySlice<Data*> arguments) const;

  // Retrieves the common device for each replica inputs (arguments[i]). The
  // common device for each replica inputs must be unique across the replicas.
  std::vector<string> GetReplicasDevices(
      const std::vector<std::vector<Data*>>& arguments) const;

  tensorflow::Tensor GetArgumentsInputs(
      tensorflow::gtl::ArraySlice<Data*> arguments, const string& device,
      tensorflow::ClientSession::FeedType* feed_inputs);

  std::vector<ExecuteContext> CreateExecuteOps(
      tensorflow::gtl::ArraySlice<const XlaComputation> computations,
      const std::vector<std::vector<Data*>>& arguments,
      tensorflow::gtl::ArraySlice<const Shape* const> output_shapes,
      std::vector<string>* devices,
      tensorflow::ClientSession::FeedType* feed_inputs);

  std::vector<ExecuteContext> CreateExecuteOps(
      const XlaComputation& computation,
      const std::vector<std::vector<Data*>>& arguments,
      const Shape* output_shape, std::vector<string>* devices,
      tensorflow::ClientSession::FeedType* feed_inputs);

  std::vector<std::shared_ptr<Data>> RunComputations(
      const std::vector<ExecuteContext>& exec_ops,
      tensorflow::gtl::ArraySlice<const XlaComputation* const> computations,
      const std::vector<string>& devices,
      const tensorflow::ClientSession::FeedType& feed_inputs);

  // Retrieves the worker,worker_host pair for a given XRT device (ie,
  // /job:tpu_worker/replica:0/task:0/device:TPU:0).
  std::pair<Worker, string> GetWorkerForXrtDevice(
      const string& xrt_device) const;

  void ReleaseHandles(tensorflow::gtl::ArraySlice<const DeviceHandle> handles);

  // Flushes all the outstanding released handles in one RPC swipe.
  void FlushReleasedHandles();

  // Function which is called at every entry into the XRT computation client
  // APIs. Performs tasks to intialize the per-call context, like flushing all
  // the accumulated handle releases, and rewinding the XRT node caches.
  void ApiCallInitialize();

  void ReleaseXrtData(XrtData* xrt_data);

  // Retrieves the mesh coordinates of a given XRT device.
  const std::vector<int>& GetDeviceMeshCoords(const string& xrt_device) const;

  tensorflow::tpu::TopologyProto InitializeAndFetchTopology(
      const string& xrt_device);

  void InitializeDevices();

  // Rewinds all the XRT node caches, marking all the cached nodes as free.
  void RewindCaches();

  // Creates an XRT graph with an XRTCompile, feeding into an XRTExecute
  // operation:
  //
  //  XRTExecute(
  //    XRTCompile(holders[0]),
  //    holders[1],
  //    holders[2]
  //  )
  //
  // With:
  //  holders[0] = XLA Computation place-holder (DT_STRING)
  //  holders[1] = xrt::XRTExecutionConfig place-holder (DT_STRING)
  //  holders[2] = Inputs for the XRTExecute (DT_INT64[])
  const CachedNode& GetCompileExecuteNode(const tensorflow::Scope& scope,
                                          const string& device);

  // Creates an XRT graph with an XRTReadLiteral operation:
  //
  //  XRTReadLiteral(
  //    holders[0]
  //  )
  //
  // With:
  //  holders[0] = The handle place-holder to be read (DT_INT64)
  const CachedNode& GetReadNode(const tensorflow::Scope& scope,
                                const string& device);

  // Creates an XRTAllocate node:
  //
  //  XRTAllocate(
  //    holders[0]
  //  )
  //
  // With:
  //  holders[0] = xrt::XLAAllocation place-holder (DT_STRING)
  const CachedNode& GetAllocateNode(const tensorflow::Scope& scope,
                                    const string& device);

  // Creates an XRTReleaseAllocationHandle node:
  //
  //  XRTReleaseAllocationHandle(
  //    holders[0]
  //  )
  //
  // With:
  //  holders[0] = To be released handle place-holder (DT_INT64)
  const CachedNode& GetReleaseAllocationHandleNode(
      const tensorflow::Scope& scope, const string& device);

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
  const CachedNode& GetSubTupleNode(const tensorflow::Scope& scope,
                                    const string& device);

  // Builds an argument vector usable in a replicated context, out of a single
  // replica argument vector. Essentially turns a [N] into a [1][N].
  static std::vector<std::vector<Data*>> BuildParallelArguments(
      tensorflow::gtl::ArraySlice<Data*> arguments);

  Options options_;
  std::map<string, std::vector<int>> device_mesh_coords_;
  std::map<string, std::unique_ptr<SessionData>> session_map_;
  std::vector<DeviceHandle> released_handles_;
  std::map<NodeCacheKey, NodeCache> node_cache_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_XRT_COMPUTATION_CLIENT_H_
