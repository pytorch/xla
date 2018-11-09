#ifndef TENSORFLOW_COMPILER_XLA_RPC_COMPUTATION_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_RPC_COMPUTATION_CLIENT_H_

#include <memory>
#include <string>
#include <vector>

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
    Data(string device, Shape device_shape)
        : device_(std::move(device)), device_shape_(std::move(device_shape)) {}

    virtual ~Data() {}

    const string& device() const { return device_; }

    const Shape& shape() const { return device_shape_; }

   private:
    string device_;
    Shape device_shape_;
  };

  static StatusOr<std::unique_ptr<ComputationClient>> Create();

  virtual ~ComputationClient() {}

  virtual std::shared_ptr<Data> TransferParameterToServer(
      const xla::Literal& literal, const string& device) = 0;

  // Executes "computation" with "arguments" and returns the result. If
  // "output_shape" isn't null, use it as a hint for the computation output
  // layout.
  virtual std::shared_ptr<Data> ExecuteComputation(
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<Data*> arguments,
      const Shape* output_shape) = 0;

  virtual std::unique_ptr<Literal> ExecuteComputationAndTransfer(
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<Data*> arguments,
      const Shape* output_shape) = 0;

  // Executes the computation in replicated mode.
  // The size of the arguments vector is the number of replicas to execute.
  // The destination devices for each replicated computation come from the
  // devices the Data objects are stored into. Within arguments[i], every Data
  // object must be coming from the same device. The optional output_shape can
  // be used to force the shape (and layout) or the computation result. Returns
  // a vector (of the same size of the arguments vector) with the results of the
  // parallel execution. The result[i] will be the result of the computation fed
  // with arguments[i].
  virtual std::vector<std::shared_ptr<Data>> ExecuteReplicated(
      const XlaComputation& computation,
      const std::vector<std::vector<Data*>>& arguments,
      const Shape* output_shape) = 0;

  // Executes the computations in parallel. Each computation must target a
  // different device. The computations[i] computation is fed with arguments[i]
  // arguments. The output_shapes[i], if not nullptr, is used to control the
  // output shape (and layout) of computations[i].
  // Returns a vector of device side Data object, with result[i] being the
  // return value of computations[i].
  virtual std::vector<std::shared_ptr<Data>> ExecuteParallel(
      tensorflow::gtl::ArraySlice<const XlaComputation> computations,
      const std::vector<std::vector<Data*>>& arguments,
      tensorflow::gtl::ArraySlice<const Shape* const> output_shapes) = 0;

  virtual StatusOr<std::vector<std::shared_ptr<Data>>> DeconstructTuple(
      const Data& data) = 0;

  virtual string GetDefaultDevice() const = 0;

  // Retrieves the ordinal number out of a device string. This is the number
  // after the last ':' character of the device string.
  static int64 GetDeviceOrdinal(const string& device);

 protected:
  // Metrics common to all client intrfaces.
  static metrics::Metric* ExecuteMetric();
  static metrics::Metric* ExecuteTransferMetric();
  static metrics::Metric* TransferMetric();
  static metrics::Metric* ExecuteReplicatedMetric();
  static metrics::Metric* ExecuteParallelMetric();
  static metrics::Metric* DeconstructTupleMetric();
  static metrics::Metric* InboundDataMetric();
  static metrics::Metric* OutboundDataMetric();
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_COMPUTATION_CLIENT_H_
