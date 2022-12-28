#ifndef XLA_CLIENT_COMPUTATION_CLIENT_H_
#define XLA_CLIENT_COMPUTATION_CLIENT_H_

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/types.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

namespace xla {

// Somehow the compiler doesn't allow type that has default member being
// used as a default parameter in a method defined in the same scope.
// Therefore, ClientExecuteOptions is defined here instead of within
// ComputationClient.
struct ClientExecuteOptions {
  bool explode_tuple{true};
};

class ComputationClient {
 public:
  class Data {
   public:
    using OpaqueHandle = int64_t;

    Data(std::string device, Shape shape)
        : device_(std::move(device)), shape_(std::move(shape)) {}

    virtual ~Data() {}

    const std::string& device() const { return device_; }

    const Shape& shape() const { return shape_; }

    virtual OpaqueHandle GetOpaqueHandle() = 0;

    virtual void Assign(const Data& data) = 0;

    virtual bool HasValue() const = 0;

   private:
    std::string device_;
    Shape shape_;
  };

  using DataPtr = std::shared_ptr<Data>;

  class Computation {
   public:
    Computation(XlaComputation computation, ProgramShape program_shape,
                std::vector<std::string> devices)
        : computation_(std::move(computation)),
          program_shape_(std::move(program_shape)),
          devices_(std::move(devices)) {}

    Computation(XlaComputation computation)
        : computation_(std::move(computation)) {
      program_shape_ = ConsumeValue(computation_.GetProgramShape());
    }

    Computation(XlaComputation computation, std::vector<std::string> devices)
        : computation_(std::move(computation)), devices_(std::move(devices)) {
      program_shape_ = ConsumeValue(computation_.GetProgramShape());
    }

    virtual ~Computation() {}

    const XlaComputation& computation() const {
      if (computation_moved_) {
        XLA_ERROR() << "Compuation has been moved\n";
      }
      return computation_;
    }

    // We don't want to make a copy when passing computation_ to the runtime.
    // Class member will be accessed as const& and `xla::XlaComputation`
    // explictly delete its const& copy constructor so we have to const cast
    // here.
    xla::XlaComputation move_computation() {
      if (computation_moved_) {
        XLA_ERROR() << "Compuation has been moved\n";
      }
      computation_moved_ = true;
      return std::move(const_cast<Computation*>(this)->computation_);
    }

    const ProgramShape& program_shape() const { return program_shape_; }

    const std::vector<std::string>& devices() const { return devices_; }

   private:
    XlaComputation computation_;
    ProgramShape program_shape_;
    std::vector<std::string> devices_;
    bool computation_moved_ = false;
  };

  using ComputationPtr = std::shared_ptr<Computation>;

  // The TensorSource provides a way for a client to populate a buffer allocated
  // by the core computation client code.
  struct TensorSource {
    // The PopulateFn accepts a dense buffer is standard array layout
    // (dim0-major) and deposits the source tensor data directly over the
    // provided buffer.
    using PopulateFn = std::function<void(const TensorSource&, void*, size_t)>;

    TensorSource() = default;
    TensorSource(Shape shape, std::string device, PopulateFn populate_fn)
        : shape(std::move(shape)),
          device(std::move(device)),
          populate_fn(std::move(populate_fn)) {}

    Shape shape;
    std::string device;
    PopulateFn populate_fn;
  };

  struct CompileInstance {
    CompileInstance() = default;
    CompileInstance(XlaComputation computation, std::string compilation_device,
                    std::vector<std::string> devices, const Shape* output_shape,
                    bool parameter_is_tupled_arguments = false,
                    bool is_sharded = false)
        : computation(std::move(computation)),
          compilation_device(std::move(compilation_device)),
          devices(std::move(devices)),
          output_shape(output_shape),
          parameter_is_tupled_arguments(parameter_is_tupled_arguments),
          is_sharded(is_sharded) {}

    XlaComputation computation;
    std::string compilation_device;
    std::vector<std::string> devices;
    const Shape* output_shape = nullptr;
    bool parameter_is_tupled_arguments;
    bool is_sharded;
  };

  struct ExecuteComputationOptions : public ClientExecuteOptions {};

  struct ExecuteReplicatedOptions : public ClientExecuteOptions {};

  struct ExecuteParallelOptions : public ClientExecuteOptions {};

  // Describes an operation to be fed to the ExecuteChained() API.
  // If the device_data member is not nullptr, this operation is a device data
  // input. Otherwise computation must not be nullptr, and represents the
  // computation to be executed. The indices of the inputs to the computation,
  // are coming from the inputs member. Since the operations fed to
  // ExecuteChained() are a valid post-order, the op_index indices listed within
  // the inputs member must be lower of the index of the current
  // ExecuteChainedOp within the post-order. If the outputs member has values,
  // the result of this ExecuteChainedOp will become an output of the
  // ExecuteChained() API, with the output_index output of this ExecuteChainedOp
  // feeding the result_index result.
  struct ExecuteChainedOp {
    struct Input {
      size_t op_index;
      absl::optional<size_t> output_index;
    };
    struct Output {
      size_t result_index;
      absl::optional<size_t> output_index;
    };

    DataPtr device_data;
    ComputationPtr computation;
    std::vector<Output> outputs;
    std::vector<Input> inputs;
  };

  struct MemoryInfo {
    int64_t kb_free = 0;
    int64_t kb_total = 0;
  };

  static std::unique_ptr<ComputationClient> Create();

  virtual ~ComputationClient() {}

  // Creates a Data object with no actual device handle in it. The device handle
  // will be populated in an asynchrounous fashion.
  virtual DataPtr CreateDataPlaceholder(std::string device, Shape shape) = 0;

  // Create DataPtr that only has dummy information which can be filled in
  // later.
  virtual std::vector<DataPtr> CreateAsyncDatas(
      absl::Span<const TensorSource> tensors) = 0;

  // Lock the DataPtr
  virtual std::vector<xla::util::ExceptionCleanup> LockAsyncDatas(
      absl::Span<const DataPtr> datas) = 0;

  // Returns data shards. We expect this to be called on PjRtShardedData to
  // retrieve the shards. If other data type is passed, it returns the input
  // wrapped inside a vector.
  virtual std::vector<DataPtr> GetDataShards(DataPtr data) = 0;

  // Transfers local tensor values to the TPU devices and fetches the handles.
  virtual std::vector<DataPtr> TransferToServer(
      absl::Span<const TensorSource> tensors) = 0;

  // Transfers local tensor values to the TPU devices and fetches the handles.
  // Update the handles associated with DataPtrs passed instead of creating new
  // datas.
  virtual void TransferToServer(absl::Span<const TensorSource> tensors,
                                absl::Span<const DataPtr> datas) = 0;

  // Transfers local sharded tensor values to the TPU devices and returns a
  // `PjRtShardedData`.
  virtual DataPtr TransferShardsToServer(
      absl::Span<const TensorSource> tensor_shards, std::string device,
      xla::Shape shape) = 0;

  // Copies `data->buffer` to `dst` device buffer.
  virtual DataPtr CopyToDevice(DataPtr data, std::string dst) = 0;

  // Reads the tensor literal values stored at TPU server sites, behind the
  // supplied handles.
  virtual std::vector<Literal> TransferFromServer(
      absl::Span<const DataPtr> handles) = 0;

  // Compiles a set of computations.
  virtual std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) = 0;

  // Executes computation with arguments and returns the result.
  // The passed device must match the common device of the arguments Data.
  // If options.explode_tuple is true, the output tuple will be decomposed into
  // its single elements.
  virtual std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      const std::string& device,
      const ExecuteComputationOptions& options =
          ExecuteComputationOptions{}) = 0;

  // Executes the computation in replicated mode.
  // The size of the arguments vector is the number of replicas to execute,
  // and it must match the size of the computation.devices() as well as the
  // devices passed as argument. The destination devices for each replicated
  // computation come from the devices the Data objects are stored into, which
  // must match the devices argument. Within arguments[i], every Data
  // object must be coming from the same device. Returns a vector (of the same
  // size of the arguments vector) with the results of the parallel execution.
  // The result[i], a vector itself, will be the result of the computation fed
  // with arguments[i]. If options.explode_tuple is true, the output tuples will
  // be decomposed into their single elements.
  virtual std::vector<std::vector<DataPtr>> ExecuteReplicated(
      const Computation& computation,
      const std::vector<std::vector<DataPtr>>& arguments,
      absl::Span<const std::string> devices,
      const ExecuteReplicatedOptions& options) = 0;

  // Executes the computations in parallel. Each computation must target a
  // different device, and the the common device of arguments[i] must match
  // devices[i]. The computations[i] computation is fed with arguments[i]
  // arguments.
  // Returns a vector of vectors of device side Data object, with result[i]
  // being the return value of computations[i]. If options.explode_tuple is
  // true, the output tuples will be decomposed into their single elements.
  virtual std::vector<std::vector<DataPtr>> ExecuteParallel(
      absl::Span<const Computation* const> computations,
      const std::vector<std::vector<DataPtr>>& arguments,
      absl::Span<const std::string> devices,
      const ExecuteParallelOptions& options) = 0;

  // Executes a serie of operations, whose results are input of other
  // operations. The ops is a valid post-order for the execution, which means
  // that the inputs of op at index I, will have to be coming from ops at index
  // lower than I. It returns a vector of device data shared pointers, one for
  // every ExecuteChainedOp marked with is_result=true, in the order they appear
  // within the ops post-order.
  virtual std::vector<DataPtr> ExecuteChained(
      absl::Span<const ExecuteChainedOp> ops, const std::string& device) = 0;

  virtual std::vector<std::vector<DataPtr>> DeconstructTuple(
      absl::Span<const DataPtr> tuples) = 0;

  // Returns a unique string which identifies the resource domain of a given
  // device. Within a resource domain, handles to device memory or compiled
  // computations can be used for all devices part of such domain.
  virtual std::string GetResourceDomain(const std::string& device) const = 0;

  virtual std::string GetDefaultDevice() const = 0;

  virtual size_t GetNumDevices() const = 0;

  virtual std::vector<std::string> GetLocalDevices() const = 0;

  virtual std::vector<std::string> GetAllDevices() const = 0;

  virtual int GetProcessIndex() const = 0;

  virtual int GetNumProcesses() const = 0;

  using DeviceAttribute =
      std::variant<std::string, int64_t, std::vector<int64_t>, float>;

  virtual const absl::flat_hash_map<std::string,
                                    xla::ComputationClient::DeviceAttribute>&
  GetDeviceAttributes(const std::string& device) = 0;

  virtual void SetReplicationDevices(
      std::shared_ptr<std::vector<std::string>> devices) = 0;

  virtual std::shared_ptr<std::vector<std::string>> GetReplicationDevices() = 0;

  virtual void SetRngSeed(size_t seed) = 0;

  virtual std::map<std::string, Metric> GetMetrics() const = 0;

  virtual MemoryInfo GetMemoryInfo(const std::string& device) = 0;

  virtual void PrepareToExit() = 0;

  // Utility API around the vector based Compile() API to compile a single
  // computation.
  ComputationPtr Compile(XlaComputation computation,
                         std::string compilation_device,
                         std::vector<std::string> devices,
                         const Shape* output_shape);

  // Retrieves the set of devices to be passed to the computation client
  // Compile() API. If the devices array is empty, a vector with the single
  // device will be returned. Otherwise a vector with the devices content will
  // be returned.
  std::vector<std::string> GetCompilationDevices(
      const std::string& device, absl::Span<const std::string> devices);

  // Run the XRT local service, this will block the caller unitl the server
  // being stopped.
  static void RunLocalService(uint64_t service_port);

  // Retrieves the ordinal number out of a device string. This is the number
  // after the last ':' character of the device string.
  static int64_t GetDeviceOrdinal(const std::string& device);

  // Returns the ComputationClient singleton.
  static ComputationClient* Get();

  static ComputationClient* GetIfInitialized();

 protected:
  // Metrics common to all client interfaces.
  static metrics::Metric* TransferToServerMetric();
  static metrics::Metric* TransferToServerTransformMetric();
  static metrics::Metric* TransferFromServerMetric();
  static metrics::Metric* CompileMetric();
  static metrics::Metric* ExecuteMetric();
  static metrics::Metric* ExecuteReplicatedMetric();
  static metrics::Metric* ExecuteParallelMetric();
  static metrics::Metric* ExecuteChainedMetric();
  static metrics::Metric* DeconstructTupleMetric();
  static metrics::Counter* CreateAsyncDataHandlesCounter();
  static metrics::Counter* CreateDataHandlesCounter();
  static metrics::Counter* ReleaseDataHandlesCounter();
  static metrics::Counter* DestroyDataHandlesCounter();
  static metrics::Metric* ReleaseDataHandlesTimeMetric();
  static metrics::Counter* CreateCompileHandlesCounter();
  static metrics::Counter* ReleaseCompileHandlesCounter();
  static metrics::Counter* DestroyCompileHandlesCounter();
  static metrics::Metric* ReleaseCompileHandlesTimeMetric();
  static metrics::Metric* InboundDataMetric();
  static metrics::Metric* OutboundDataMetric();
};

}  // namespace xla

#endif  // XLA_CLIENT_COMPUTATION_CLIENT_H_
