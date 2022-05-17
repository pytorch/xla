#ifndef XLA_CLIENT_PJRT_COMPUTATION_CLIENT_H_
#define XLA_CLIENT_PJRT_COMPUTATION_CLIENT_H_

#include <cstdint>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

namespace xla {

class PjRtComputationClient : public ComputationClient {
 public:
  PjRtComputationClient();

  DataPtr CreateDataPlaceholder(std::string device, Shape shape) override;

  std::vector<DataPtr> TransferToServer(
      absl::Span<const TensorSource> tensors) override;

  std::vector<Literal> TransferFromServer(
      absl::Span<const DataPtr> handles) override;

  std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) override;

  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      const std::string& device,
      const ExecuteComputationOptions& options) override;

  size_t GetNumDevices() const override;

  std::string GetDefaultDevice() const override;

  std::vector<std::string> GetLocalDevices() const override;

  std::vector<std::string> GetAllDevices() const override;

  void SetReplicationDevices(
      std::shared_ptr<std::vector<std::string>> devices) override;

  std::shared_ptr<std::vector<std::string>> GetReplicationDevices() override;

  void PrepareToExit() override { return; };

  // NOT IMPLEMENTED

  void TransferToServer(absl::Span<const TensorSource> tensors,
                        absl::Span<const DataPtr> datas) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  std::vector<DataPtr> CreateAsyncDatas(
      absl::Span<const TensorSource> tensors) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  std::vector<xla::util::ExceptionCleanup> LockAsyncDatas(
      absl::Span<const DataPtr> datas) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  std::vector<std::vector<DataPtr>> DeconstructTuple(
      absl::Span<const DataPtr> tuples) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  std::vector<std::vector<DataPtr>> ExecuteReplicated(
      const Computation& computation,
      const std::vector<std::vector<DataPtr>>& arguments,
      absl::Span<const std::string> devices,
      const ExecuteReplicatedOptions& options) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  std::vector<std::vector<DataPtr>> ExecuteParallel(
      absl::Span<const Computation* const> computations,
      const std::vector<std::vector<DataPtr>>& arguments,
      absl::Span<const std::string> devices,
      const ExecuteParallelOptions& options) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  std::vector<DataPtr> ExecuteChained(absl::Span<const ExecuteChainedOp> ops,
                                      const std::string& device) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  std::string GetResourceDomain(const std::string& device) const override {
    // TODO(wcromar): return a meaningful value
    return "getresourcedomainplaceholder";
  };

  void SetRngSeed(size_t seed) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  std::map<std::string, Metric> GetMetrics() const override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  MemoryInfo GetMemoryInfo(const std::string& device) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

 private:
  std::shared_ptr<PjRtClient> client_;
  std::unordered_map<std::string, xla::PjRtDevice* const> string_to_device_;
  std::shared_ptr<std::vector<std::string>> replication_devices_;

  xla::PjRtDevice* StringToPjRtDevice(const std::string& device);

  struct PjRtData : public Data {
    PjRtData(std::string device, Shape device_shape)
        : Data(std::move(device), std::move(device_shape)) {}

    PjRtData(std::string device, Shape device_shape,
             std::shared_ptr<PjRtBuffer> buffer)
        : Data(std::move(device), std::move(device_shape)), buffer(buffer) {}

    void* get_handle() const {
      return buffer->AcquireExternalReference()
          .ValueOrDie()
          ->OpaqueDeviceMemoryDataPointer();
    };
    OpaqueHandle GetOpaqueHandle() override {
      return reinterpret_cast<std::uintptr_t>(get_handle());
    };
    void Assign(const Data& data) override;
    bool HasValue() const override {
      return buffer != nullptr && !buffer->IsDeleted();
    };

    std::shared_ptr<PjRtBuffer> buffer;
  };

  struct PjRtComputation : public Computation {
    PjRtComputation(XlaComputation computation, ProgramShape program_shape,
                    std::vector<std::string> devices,
                    std::unique_ptr<xla::PjRtExecutable> executable)
        : Computation(std::move(computation), std::move(program_shape),
                      std::move(devices)),
          executable(std::move(executable)) {}

    std::unique_ptr<xla::PjRtExecutable> executable;
  };
};

}  // namespace xla
#endif  // XLA_CLIENT_XRT_COMPUTATION_CLIENT_H_
