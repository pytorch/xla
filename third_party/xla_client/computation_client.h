#ifndef TENSORFLOW_COMPILER_XLA_RPC_COMPUTATION_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_RPC_COMPUTATION_CLIENT_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace xla {

class ComputationClient {
 public:
  class Data {
   public:
    Data(string device, Shape shape)
        : unique_id_(GetNextDataId()),
          device_(std::move(device)),
          shape_(std::move(shape)) {}

    virtual ~Data() {}

    int64 unique_id() const { return unique_id_; }

    const string& device() const { return device_; }

    const Shape& shape() const { return shape_; }

   private:
    int64 unique_id_ = 0;
    string device_;
    Shape shape_;
  };

  class Computation {
   public:
    Computation(XlaComputation computation, ProgramShape program_shape,
                std::vector<string> devices)
        : computation_(std::move(computation)),
          program_shape_(std::move(program_shape)),
          devices_(std::move(devices)) {}

    virtual ~Computation() {}

    const XlaComputation& computation() const { return computation_; }

    const ProgramShape& program_shape() const { return program_shape_; }

    const std::vector<string>& devices() const { return devices_; }

   private:
    XlaComputation computation_;
    ProgramShape program_shape_;
    std::vector<string> devices_;
  };

  // The TensorSource provides a way for a client to populate a buffer allocated
  // by the core computation client code.
  struct TensorSource {
    // The PopulateFn accepts a dense buffer is standard array layout
    // (dim0-major) and deposits the source tensor data directly over the
    // provided buffer.
    using PopulateFn = std::function<void(const TensorSource&, void*, size_t)>;

    TensorSource() = default;
    TensorSource(Shape shape, string device, PopulateFn populate_fn)
        : shape(std::move(shape)),
          device(std::move(device)),
          populate_fn(std::move(populate_fn)) {}

    Shape shape;
    string device;
    PopulateFn populate_fn;
  };

  struct CompileInstance {
    CompileInstance() = default;
    CompileInstance(XlaComputation computation, std::vector<string> devices,
                    const Shape* output_shape)
        : computation(std::move(computation)),
          devices(std::move(devices)),
          output_shape(output_shape) {}

    XlaComputation computation;
    std::vector<string> devices;
    const Shape* output_shape = nullptr;
  };

  struct ExecuteOptions {
    bool explode_tuple = true;
  };

  struct ExecuteComputationOptions : public ExecuteOptions {};

  struct ExecuteReplicatedOptions : public ExecuteOptions {};

  struct ExecuteParallelOptions : public ExecuteOptions {};

  static StatusOr<std::unique_ptr<ComputationClient>> Create();

  virtual ~ComputationClient() {}

  // Transfers local tensor values to the TPU servers and fetches the handles.
  virtual std::vector<std::shared_ptr<Data>> TransferToServer(
      tensorflow::gtl::ArraySlice<const TensorSource> tensors) = 0;

  // Reads the tensor literal values stored at TPU server sites, behind the
  // supplied handles.
  virtual std::vector<Literal> TransferFromServer(
      tensorflow::gtl::ArraySlice<const std::shared_ptr<Data>> handles) = 0;

  // Compiles a set of computations.
  virtual std::vector<std::shared_ptr<Computation>> Compile(
      std::vector<CompileInstance> instances) = 0;

  // Executes computation with arguments and returns the result.
  // The passed device must match the common device of the arguments Data.
  // If options.explode_tuple is true, the output tuple will be decomposed into
  // its single elements.
  virtual std::vector<std::shared_ptr<Data>> ExecuteComputation(
      const Computation& computation,
      tensorflow::gtl::ArraySlice<Data*> arguments, const string& device,
      const ExecuteComputationOptions& options) = 0;

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
  virtual std::vector<std::vector<std::shared_ptr<Data>>> ExecuteReplicated(
      const Computation& computation,
      const std::vector<std::vector<Data*>>& arguments,
      tensorflow::gtl::ArraySlice<const string> devices,
      const ExecuteReplicatedOptions& options) = 0;

  // Executes the computations in parallel. Each computation must target a
  // different device, and the the common device of arguments[i] must match
  // devices[i]. The computations[i] computation is fed with arguments[i]
  // arguments.
  // Returns a vector of vectors of device side Data object, with result[i]
  // being the return value of computations[i]. If options.explode_tuple is
  // true, the output tuples will be decomposed into their single elements.
  virtual std::vector<std::vector<std::shared_ptr<Data>>> ExecuteParallel(
      tensorflow::gtl::ArraySlice<const Computation* const> computations,
      const std::vector<std::vector<Data*>>& arguments,
      tensorflow::gtl::ArraySlice<const string> devices,
      const ExecuteParallelOptions& options) = 0;

  virtual std::vector<std::vector<std::shared_ptr<Data>>> DeconstructTuple(
      tensorflow::gtl::ArraySlice<const std::shared_ptr<Data>> tuples) = 0;

  virtual string GetDefaultDevice() const = 0;

  // Utility API around the vector based Compile() API to compile a single
  // computation.
  std::shared_ptr<Computation> Compile(XlaComputation computation,
                                       std::vector<string> devices,
                                       const Shape* output_shape);

  // Retrieves the ordinal number out of a device string. This is the number
  // after the last ':' character of the device string.
  static int64 GetDeviceOrdinal(const string& device);

  // Returns the ComputationClient singleton.
  static ComputationClient* Get();

 protected:
  // Generates a new unique ID for a Data object.
  static int64 GetNextDataId();

  // Metrics common to all client intrfaces.
  static metrics::Metric* TransferToServerMetric();
  static metrics::Metric* TransferFromServerMetric();
  static metrics::Metric* CompileMetric();
  static metrics::Metric* ExecuteMetric();
  static metrics::Metric* ExecuteReplicatedMetric();
  static metrics::Metric* ExecuteParallelMetric();
  static metrics::Metric* DeconstructTupleMetric();
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

#endif  // TENSORFLOW_COMPILER_XLA_RPC_COMPUTATION_CLIENT_H_
