#ifndef XLA_CLIENT_XLA_COMPUTATION_CLIENT_H_
#define XLA_CLIENT_XLA_COMPUTATION_CLIENT_H_

#include <sys/syscall.h>

#include <memory>
#include <sstream>
#include <stdexcept>

#include "tensorflow/compiler/xla/service_interface.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include "tensorflow/compiler/xla/xla_client/proxy_client_util.h"
#include "tensorflow/compiler/xla/xla_client/proxy_name.h"

namespace xla {

/**
 * @brief Computation Client which maps to the xla::ServiceInterface
 *        interface, allthough the expectation is that the service
 *        behaves more like the XRT interface as it applies to handles,
 *        which appears to differ somewhat from the "true" xla grpc
 *        service which can be built in the source tree.
 */
class XlaComputationClient : public ComputationClient {
  typedef ComputationClient Super;
  using DeviceHandle = XrtComputationClient::DeviceHandle;

public:
  explicit XlaComputationClient(std::shared_ptr<xla::ServiceInterface> service);
  virtual ~XlaComputationClient() override;

  // Creates a Data object with no actual device handle in it. The device handle
  // will be populated in an asynchrounous fashion.
  DataPtr CreateDataPlaceholder(std::string device, Shape shape) override;

  // Transfers local tensor values to the TPU servers and fetches the handles.
  std::vector<DataPtr>
  TransferToServer(absl::Span<const TensorSource> tensors) override;

  // Reads the tensor literal values stored at TPU server sites, behind the
  // supplied handles.
  std::vector<Literal>
  TransferFromServer(absl::Span<const DataPtr> handles) override;

  // Compiles a set of computations.
  std::vector<ComputationPtr>
  Compile(std::vector<CompileInstance> instances) override;

  // Executes computation with arguments and returns the result.
  // The passed device must match the common device of the arguments Data.
  // If options.explode_tuple is true, the output tuple will be decomposed into
  // its single elements.
  std::vector<DataPtr>
  ExecuteComputation(const Computation &computation,
                     absl::Span<const DataPtr> arguments,
                     const std::string &device,
                     const ExecuteComputationOptions &options) override;

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
  std::vector<std::vector<DataPtr>>
  ExecuteReplicated(const Computation &computation,
                    const std::vector<std::vector<DataPtr>> &arguments,
                    absl::Span<const std::string> devices,
                    const ExecuteReplicatedOptions &options) override {
    // NOT IMPLEMENTED
    assert(false);
    return {};
  }

  // Executes the computations in parallel. Each computation must target a
  // different device, and the the common device of arguments[i] must match
  // devices[i]. The computations[i] computation is fed with arguments[i]
  // arguments.
  // Returns a vector of vectors of device side Data object, with result[i]
  // being the return value of computations[i]. If options.explode_tuple is
  // true, the output tuples will be decomposed into their single elements.
  std::vector<std::vector<DataPtr>>
  ExecuteParallel(absl::Span<const Computation *const> computations,
                  const std::vector<std::vector<DataPtr>> &arguments,
                  absl::Span<const std::string> devices,
                  const ExecuteParallelOptions &options) override {
    // NOT IMPLEMENTED
    assert(false);
    return {};
  }

  // Executes a serie of operations, whose results are input of other
  // operations. The ops is a valid post-order for the execution, which means
  // that the inputs of op at index I, will have to be coming from ops at index
  // lower than I. It returns a vector of device data shared pointers, one for
  // every ExecuteChainedOp marked with is_result=true, in the order they appear
  // within the ops post-order.
  std::vector<DataPtr> ExecuteChained(absl::Span<const ExecuteChainedOp> ops,
                                      const std::string &device) override {
    // NOT IMPLEMENTED
    assert(false);
    return {};
  }

  std::vector<std::vector<DataPtr>>
  DeconstructTuple(absl::Span<const DataPtr> tuples) override {
    // NOT IMPLEMENTED
    assert(false);
    return {};
  }

  MemoryInfo GetMemoryInfo(const std::string &device) override {
    // Just return junk data since xla::ServiceInterface
    // doesn't offer this information
    return MemoryInfo{.kb_free = 1024, .kb_total = 1024};
  }

  void PrepareToExit() override;

  void ReleaseDataByHandle(const std::string &device, int64 handle) override;

  //===================================================================
  //
  // This section is more "global" than a single ComputationClient
  // instance, so may need to be in a separate interface in order
  // to morecleanly allow multiple instances of standalone
  // ComputationClient objects.
  //
  //===================================================================

  // Returns a unique string which identifies the resource domain of a given
  // device. Within a resource domain, handles to device memory or compiled
  // computations can be used for all devices part of such domain.
  std::string GetResourceDomain(const std::string &device) const override {
    // NOT IMPLEMENTED/NOT NEEDED
    assert(false);
    return {};
  }

  std::string GetDefaultDevice() const override {
    // NOT IMPLEMENTED/NOT NEEDED
    assert(false);
    return {};
  }

  size_t GetNumDevices() const override {
    // NOT IMPLEMENTED/NOT NEEDED
    assert(false);
    return {};
  }

  std::vector<std::string> GetLocalDevices() const override {
    // NOT IMPLEMENTED/NOT NEEDED
    assert(false);
    return {};
  }

  std::vector<std::string> GetAllDevices() const override {
    // NOT IMPLEMENTED/NOT NEEDED
    assert(false);
    return {};
  }

  void SetReplicationDevices(
      std::shared_ptr<std::vector<std::string>> devices) override {
    // NOT IMPLEMENTED/NOT NEEDED
    assert(false);
  }

  std::shared_ptr<std::vector<std::string>> GetReplicationDevices() override {
    // NOT IMPLEMENTED/NOT NEEDED
    assert(false);
    return nullptr;
  }

  void SetRngSeed(size_t seed) override {
    // NOT IMPLEMENTED/NOT NEEDED
  }

  std::map<std::string, Metric> GetMetrics() const override {
    // NOT IMPLEMENTED/NOT NEEDED
    assert(false);
    return {};
  }

  //
  // TODO: REMOVE ME
  //
  static std::shared_ptr<xla::ServiceInterface>
  GetXlaClient(const std::string &device, bool create = true);

  ComputationClient::DataPtr TransferLiteralToServer(const std::string &device,
                                                     const Literal &literal);

  static std::shared_ptr<xla::ServiceInterface>
  CreateServiceClient(const std::string &address);

  // TEMPORARY
  std::shared_ptr<xla::ServiceInterface> GetXlaClient() { return service_; }

private:
  xla::DeviceHandle GetDeviceHandle(const std::string &device);

  // Asynchronous data release mechanism
  // This mechanism is common among mnost interface types and
  // should possibly be shared in a base or orthogonal class

  virtual xla::HloModuleProto
  PreProcessHlo(xla::HloModuleProto &&hlo_module_proto);

  void StartHandleReleaser();
  void HandleReleaser();
  void ReleaseHandleAsync(int64 handle, const std::string &device,
                          std::vector<DeviceHandle> *handles);
  void ReleaseComputation(const std::string &compilation_device, int64 handle);
  void ReleaseHandles(
      std::vector<DeviceHandle> *handles,
      const std::function<std::shared_ptr<xla::ServiceInterface>(
          const std::shared_ptr<xla::ServiceInterface> &,
          const tensorflow::Scope &, const std::string &)> &op_generator,
      metrics::Metric *timed_metric, metrics::Counter *destroy_counter);

  std::shared_ptr<xla::ServiceInterface> service_;

  std::mutex task_lock_;
  std::unique_ptr<util::TriggeredTask> triggered_task_;
  std::vector<DeviceHandle> released_data_handles_;
  std::vector<DeviceHandle> released_compile_handles_;
};

class XlaComputationClientFactory : public ComputationClientFactory {
public:
  explicit XlaComputationClientFactory(std::string device, bool create_proxy)
      : device_(std::move(device)) {
    assert(create_proxy || !ProxyName::is_proxy_device_name(device));
  }

  std::unique_ptr<ComputationClient>
  Create(OptionsType options,
         std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto,
         XrtLocalService *service) override {
    return Create();
  }

  std::unique_ptr<ComputationClient> Create() override {
    /// TODO: remove knowledge of a "proxy name"
    return std::make_unique<XlaComputationClient>(
        XlaComputationClient::GetXlaClient(
            ProxyName::proxy_device_name(device_)));
  }

  tensorflow::tpu::TopologyProto
  InitializeAndFetchTopology(const std::string &job, int task_no,
                             const std::string &worker_host_port,
                             const tensorflow::ConfigProto &config) override {
    // No meaningful topology for a basic xla client
    return tensorflow::tpu::TopologyProto();
  }

private:
  const std::string device_;
};

} // namespace xla

#endif // XLA_CLIENT_XLA_COMPUTATION_CLIENT_H_