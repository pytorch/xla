#pragma once

#include <ATen/core/Tensor.h>

#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/nnc_computation_client.h"

namespace xla {

class GenericComputationXla : public GenericComputation {
 public:
  GenericComputationXla(XlaComputation computation)
      : computation_(std::move(computation)) {}

  StatusOr<ProgramShape> GetProgramShape() const override {
    return computation_.GetProgramShape();
  }

  const XlaComputation& computation() const { return computation_; }

 private:
  XlaComputation computation_;
};

namespace compiler {

class NNCComputationClient : public ComputationClient {
 public:
  struct NNCData : public Data {
    NNCData(const at::Tensor& data, Shape shape, std::string device)
        : Data(std::move(device), std::move(shape)),
          data_(xla::NNCComputationClient::HardwareDeviceType() == at::kCUDA
                    ? data.cuda()
                    : data) {}

    NNCData(Shape shape, std::string device)
        : Data(std::move(device), std::move(shape)) {}

    OpaqueHandle GetOpaqueHandle() override {
      return reinterpret_cast<int64>(this);
    }

    void Assign(const Data& data) override {
      data_ = static_cast<const NNCData&>(data).data_;
    }

    bool HasValue() const override { return data_.defined(); }

    at::Tensor data_;
  };

  DataPtr CreateDataPlaceholder(std::string device, Shape shape) override;

  std::vector<DataPtr> TransferToServer(
      absl::Span<const TensorSource> tensors) override {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<Literal> TransferFromServer(
      absl::Span<const DataPtr> handles) override {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) override;

  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      const std::string& device,
      const ExecuteComputationOptions& options) override;

  std::vector<std::vector<DataPtr>> ExecuteReplicated(
      const Computation& computation,
      const std::vector<std::vector<DataPtr>>& arguments,
      absl::Span<const std::string> devices,
      const ExecuteReplicatedOptions& options) override {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<std::vector<DataPtr>> ExecuteParallel(
      absl::Span<const Computation* const> computations,
      const std::vector<std::vector<DataPtr>>& arguments,
      absl::Span<const std::string> devices,
      const ExecuteParallelOptions& options) override {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<DataPtr> ExecuteChained(absl::Span<const ExecuteChainedOp> ops,
                                      const std::string& device) override {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<std::vector<DataPtr>> DeconstructTuple(
      absl::Span<const DataPtr> tuples) override {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  std::string GetResourceDomain(const std::string& device) const override;

  std::string GetDefaultDevice() const override;

  size_t GetNumDevices() const override { return 1; }

  std::vector<std::string> GetLocalDevices() const override;

  std::vector<std::string> GetAllDevices() const override;

  void SetReplicationDevices(
      std::shared_ptr<std::vector<std::string>> devices) override;

  std::shared_ptr<std::vector<std::string>> GetReplicationDevices() override;

  void SetRngSeed(size_t seed) override {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  std::map<std::string, Metric> GetMetrics() const override { return {}; }

  MemoryInfo GetMemoryInfo(const std::string& device) override {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  void PrepareToExit() override;
};

ComputationClient* NNCGet();

ComputationClient* NNCGetIfInitialized();

}  // namespace compiler
}  // namespace xla
