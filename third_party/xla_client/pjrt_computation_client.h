#ifndef XLA_CLIENT_PJRT_COMPUTATION_CLIENT_H_
#define XLA_CLIENT_PJRT_COMPUTATION_CLIENT_H_

#include <cstdint>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"

namespace xla {

class PjRtComputationClient : public ComputationClient {
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
    bool HasValue() const override { return get_handle() != nullptr; };

    std::shared_ptr<PjRtBuffer> buffer;
  };

  struct PjRtComputation : public Computation {
    PjRtComputation(PjRtClient* self, XlaComputation computation,
                    ProgramShape program_shape,
                    std::vector<std::string> devices,
                    xla::CompileOptions options)
        : Computation(std::move(computation), std::move(program_shape),
                      std::move(devices)),
          executable(self->Compile(this->computation(), options).ValueOrDie()) {
    }

    // TODO: something other than what I'm doing here
    int64_t get_handle() const {
      return executable->SizeOfGeneratedCodeInBytes();
    }

    std::unique_ptr<xla::PjRtExecutable> executable;
  };

 public:
  struct Options {};

  PjRtComputationClient(
      Options options/*,
      // TODO: fix
      std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto*/);

  DataPtr CreateDataPlaceholder(std::string device, Shape shape) override;

  std::vector<DataPtr> TransferToServer(
      absl::Span<const TensorSource> tensors) override;

  void TransferToServer(absl::Span<const TensorSource> tensors,
                        absl::Span<const DataPtr> datas) override {
    throw std::logic_error(__FUNCTION__);
  };

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

  std::shared_ptr<std::vector<std::string>> GetReplicationDevices() override;

  void PrepareToExit() override { return; };

  // NOT IMPLEMENTED

  std::vector<DataPtr> CreateAsyncDatas(
      absl::Span<const TensorSource> tensors) override {
    throw std::logic_error(__FUNCTION__);
  };

  std::vector<xla::util::ExceptionCleanup> LockAsyncDatas(
      absl::Span<const DataPtr> datas) override {
    throw std::logic_error(__FUNCTION__);
  };

  std::vector<std::vector<DataPtr>> DeconstructTuple(
      absl::Span<const DataPtr> tuples) override {
    throw std::logic_error(__FUNCTION__);
  };

  std::vector<std::vector<DataPtr>> ExecuteReplicated(
      const Computation& computation,
      const std::vector<std::vector<DataPtr>>& arguments,
      absl::Span<const std::string> devices,
      const ExecuteReplicatedOptions& options) override {
    throw std::logic_error(__FUNCTION__);
  };

  std::vector<std::vector<DataPtr>> ExecuteParallel(
      absl::Span<const Computation* const> computations,
      const std::vector<std::vector<DataPtr>>& arguments,
      absl::Span<const std::string> devices,
      const ExecuteParallelOptions& options) override {
    throw std::logic_error(__FUNCTION__);
  };

  std::vector<DataPtr> ExecuteChained(absl::Span<const ExecuteChainedOp> ops,
                                      const std::string& device) override {
    throw std::logic_error(__FUNCTION__);
  };

  std::string GetResourceDomain(const std::string& device) const override {
    return "getresourcedomainplaceholder";
  };

  void SetReplicationDevices(
      std::shared_ptr<std::vector<std::string>> devices) override {
    throw std::logic_error(__FUNCTION__);
  };

  void SetRngSeed(size_t seed) override {
    throw std::logic_error(__FUNCTION__);
  };

  std::map<std::string, Metric> GetMetrics() const override {
    throw std::logic_error(__FUNCTION__);
  };

  MemoryInfo GetMemoryInfo(const std::string& device) override {
    throw std::logic_error(__FUNCTION__);
  };

 private:
  std::unique_ptr<PjRtClient> client;
};

}  // namespace xla
#endif  // XLA_CLIENT_XRT_COMPUTATION_CLIENT_H_
