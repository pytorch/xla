#ifndef XLA_CLIENT_XLA_COMPUTATION_PROXY_H_
#define XLA_CLIENT_XLA_COMPUTATION_PROXY_H_

#include <sys/syscall.h>

#include <memory>
#include <sstream>
#include <stdexcept>

#include "tensorflow/compiler/xla/service_interface.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include "tensorflow/compiler/xla/xla_client/computation_client_manager.h"

namespace xla {

class ProxyClientInfo;
class GlobalDataHandleMapper;

/**
 * @brief Manage a proxy device in which "approved" computations
 *        shall execute and to which will be forwarded.
 */
class ProxyComputationClient : public XrtComputationClient {
  typedef XrtComputationClient Super;

 public:
  /**
   * @brief Create ProxyComputationClient object
   */
  ProxyComputationClient(
      Options options,
      std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto,
      XrtLocalService *service = nullptr);

  /**
   * @brief Destroy the ProxyComputationClient object
   */
  ~ProxyComputationClient() override = default;

  // Creates a Data object with no actual device handle in it. The device handle
  // will be populated in an asynchrounous fashion.
  DataPtr CreateDataPlaceholder(std::string device, Shape shape) override;

  // Transfers local tensor values to the TPU servers and fetches the handles.
  std::vector<DataPtr> TransferToServer(
      absl::Span<const TensorSource> tensors) override;

  // Reads the tensor literal values stored at TPU server sites, behind the
  // supplied handles.
  std::vector<Literal> TransferFromServer(
      absl::Span<const DataPtr> handles) override;

  // Compiles a set of computations.
  std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) override;

  // Executes computation with arguments and returns the result.
  // The passed device must match the common device of the arguments Data.
  // If options.explode_tuple is true, the output tuple will be decomposed into
  // its single elements.
  std::vector<DataPtr> ExecuteComputation(
      const Computation &computation, absl::Span<const DataPtr> arguments,
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
  std::vector<std::vector<DataPtr>> ExecuteReplicated(
      const Computation &computation,
      const std::vector<std::vector<DataPtr>> &arguments,
      absl::Span<const std::string> devices,
      const ExecuteReplicatedOptions &options) override;

  // Executes the computations in parallel. Each computation must target a
  // different device, and the the common device of arguments[i] must match
  // devices[i]. The computations[i] computation is fed with arguments[i]
  // arguments.
  // Returns a vector of vectors of device side Data object, with result[i]
  // being the return value of computations[i]. If options.explode_tuple is
  // true, the output tuples will be decomposed into their single elements.
  std::vector<std::vector<DataPtr>> ExecuteParallel(
      absl::Span<const Computation *const> computations,
      const std::vector<std::vector<DataPtr>> &arguments,
      absl::Span<const std::string> devices,
      const ExecuteParallelOptions &options) override;

  // Executes a serie of operations, whose results are input of other
  // operations. The ops is a valid post-order for the execution, which means
  // that the inputs of op at index I, will have to be coming from ops at index
  // lower than I. It returns a vector of device data shared pointers, one for
  // every ExecuteChainedOp marked with is_result=true, in the order they appear
  // within the ops post-order.
  std::vector<DataPtr> ExecuteChained(absl::Span<const ExecuteChainedOp> ops,
                                      const std::string &device) override;

  //
  // Other APIs
  //
  static tensorflow::tpu::TopologyProto InitializeAndFetchTopology(
      const std::string &job, int task_no, const std::string &worker_host_port,
      const tensorflow::ConfigProto &config);

  static ProxyComputationClient *Get() {
    return dynamic_cast<ProxyComputationClient *>(Super::Get());
  }

  /**
   * @brief Returns 'true' if
   */
  static bool IsEnabled();

  /**
   * @brief Returns 'true' if an object of this class has been
   *        created in this current process (or a forked parent)
   */
  static bool IsInitialized();

  void PrepareToExit() override;

 private:
  /**
   * @brief
   */
  std::vector<ComputationClient::DataPtr> MoveDataBetweenDevices(
      const std::vector<ComputationClient::DataPtr> &source_data,
      const std::string &to_device, bool release_from_source);

  /**
   * @brief For the given tensors, assure that the data for the given tensors
   *        exits on both the normal device and the proxy device (proxy
   *        for the given device) and add the necessary handle mapping.
   *        I fthe data already exists on both devices, then no movement
   *        will occur.
   * @param tensors Tensors we wish to be copied to 'device' as necessary
   * @param device The device where the copy should be placed
   * @param in_place When 'true', modify the associated XrtData object's
   *        handle to reference the handle on the destination device
   * @return  DataPtr for the data on the destination device
   */
  std::vector<DataPtr> NormalizeDataToDevice(absl::Span<const DataPtr> tensors,
                                             const std::string &device,
                                             bool in_place);

  bool ShouldCloneDataForDevice(const std::string &device) const;
  bool IsProxyExecutable(uint64_t executable_handle) const;
  void AddProxyExecutable(uint64_t executable_handle);

  /**
   * @brief Overridden in order to capture XrtComputationClient's releases
   *        in order to remove the mapping to the proxy device's data handles.
   */
  void ReleaseXrtData(const std::string &device, int64 handle) override;

  /**
   * @brief Currently maintains high-speed clients
   *        (multisupports multiple devices).
   *        Ideally, this would also hold the "local" "non-proxy"
   *        device as well, however that would require the
   *        "heavy" init section to be separated from XrtComputationClient
   *        and so it remains as a base class in order to fqacilitate
   *        maintainability.
   *        Another approach would be to enable_shared_from_this() on the
   *        base class with a weak_ptr and hold it in client_manager_ as just
   *        another ComputationClient, unbeknownst to this class that it
   *        indeed derives from it.
   */
  ComputationClientManager client_manager_;

  /**
   * @brief Manages the mapping of resource handles which are aliased
   *        between multiple devices
   */
  std::shared_ptr<GlobalDataHandleMapper> data_mapper_;

  /**
   * @brief Maintain a set of executable known to be
   *        executed on the proxy device
   */
  mutable std::mutex proxy_executable_set_mtx_;
  std::unordered_set<uint64_t> proxy_executable_set_;
};

}  // namespace xla

#endif  /// XLA_CLIENT_XLA_COMPUTATION_PROXY_H_