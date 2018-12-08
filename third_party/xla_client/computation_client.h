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
        : device_(std::move(device)), shape_(std::move(shape)) {}

    virtual ~Data() {}

    const string& device() const { return device_; }

    const Shape& shape() const { return shape_; }

   private:
    string device_;
    Shape shape_;
  };

  struct LiteralDevice {
    LiteralDevice() = default;
    LiteralDevice(Literal literal, string device)
        : literal(std::move(literal)), device(std::move(device)) {}
    LiteralDevice(std::function<Literal()> literal_fn, string device)
        : literal_fn(std::move(literal_fn)), device(std::move(device)) {}

    const Literal& GetLiteral(Literal* tmp) const {
      if (literal) {
        return *literal;
      }
      *tmp = literal_fn();
      return *tmp;
    }

    absl::optional<Literal> literal;
    std::function<Literal()> literal_fn;
    string device;
  };

  static StatusOr<std::unique_ptr<ComputationClient>> Create();

  virtual ~ComputationClient() {}

  // Signals the ComputationClient object to flush all the lazy resource
  // releases it has accumulated so far.
  virtual void FlushLazyReleases() = 0;

  // Forcefully releases the device memory handles, and returns the number of
  // handles which has been effectively released.
  // This API should only be called once it is known that the other referrers of
  // the handles will not be using them anymore (like in the really late exit
  // path of the application). Using the handles after such API call can caused
  // undefined behavior and/or crashes.
  virtual size_t ForceReleaseHandles(
      tensorflow::gtl::ArraySlice<const std::shared_ptr<Data>> handles) = 0;

  // Transfers local tensor literal values to the TPU servers and fetches the
  // handles.
  virtual std::vector<std::shared_ptr<Data>> TransferToServer(
      tensorflow::gtl::ArraySlice<const LiteralDevice> literals) = 0;

  // Reads the tensor literal values stored at TPU server sites, behind the
  // supplied handles.
  virtual std::vector<Literal> TransferFromServer(
      tensorflow::gtl::ArraySlice<const std::shared_ptr<Data>> handles) = 0;

  // Executes "computation" with "arguments" and returns the result. If
  // "output_shape" isn't null, use it as a hint for the computation output
  // layout. The passed device must match the common device of the arguments
  // Data.
  virtual std::shared_ptr<Data> ExecuteComputation(
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<Data*> arguments, const string& device,
      const Shape* output_shape) = 0;

  // Executes the computation in replicated mode.
  // The size of the arguments vector is the number of replicas to execute.
  // The destination devices for each replicated computation come from the
  // devices the Data objects are stored into, which must match the passed in
  // devices. The reason of the devices argument is due to the fact that the
  // caller will expect a given computation to happen in one device, but such
  // computation has no parameters. Within arguments[i], every Data object must
  // be coming from the same device. The optional output_shape can be used to
  // force the shape (and layout) or the computation result. Returns a vector
  // (of the same size of the arguments vector) with the results of the parallel
  // execution. The result[i] will be the result of the computation fed with
  // arguments[i].
  virtual std::vector<std::shared_ptr<Data>> ExecuteReplicated(
      const XlaComputation& computation,
      const std::vector<std::vector<Data*>>& arguments,
      tensorflow::gtl::ArraySlice<const string> devices,
      const Shape* output_shape) = 0;

  // Executes the computations in parallel. Each computation must target a
  // different device, the the common device of arguments[i] must match
  // devices[i]. The computations[i] computation is fed with arguments[i]
  // arguments. The output_shapes[i], if not nullptr, is used to control the
  // output shape (and layout) of computations[i].
  // Returns a vector of device side Data object, with result[i] being the
  // return value of computations[i].
  virtual std::vector<std::shared_ptr<Data>> ExecuteParallel(
      tensorflow::gtl::ArraySlice<const XlaComputation> computations,
      const std::vector<std::vector<Data*>>& arguments,
      tensorflow::gtl::ArraySlice<const string> devices,
      tensorflow::gtl::ArraySlice<const Shape* const> output_shapes) = 0;

  virtual std::vector<std::vector<std::shared_ptr<Data>>> DeconstructTuple(
      tensorflow::gtl::ArraySlice<const std::shared_ptr<Data>> tuples) = 0;

  virtual string GetDefaultDevice() const = 0;

  // Retrieves the ordinal number out of a device string. This is the number
  // after the last ':' character of the device string.
  static int64 GetDeviceOrdinal(const string& device);

 protected:
  // Metrics common to all client intrfaces.
  static metrics::Metric* TransferToServerMetric();
  static metrics::Metric* TransferFromServerMetric();
  static metrics::Metric* ExecuteMetric();
  static metrics::Metric* ExecuteReplicatedMetric();
  static metrics::Metric* ExecuteParallelMetric();
  static metrics::Metric* DeconstructTupleMetric();
  static metrics::Counter* CreateHandlesCounter();
  static metrics::Counter* ReleaseHandlesCounter();
  static metrics::Counter* DestroyHandlesCounter();
  static metrics::Metric* ReleaseHandlesTimeMetric();
  static metrics::Metric* InboundDataMetric();
  static metrics::Metric* OutboundDataMetric();
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_COMPUTATION_CLIENT_H_
