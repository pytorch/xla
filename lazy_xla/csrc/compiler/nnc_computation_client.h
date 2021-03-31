#pragma once

#include <ATen/core/Tensor.h>

#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/computation_client/nnc_computation_client.h"
#include "lazy_xla/csrc/compiler/helpers.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"

namespace xla {

class GenericComputationXla : public lazy_tensors::GenericComputation {
 public:
  GenericComputationXla(XlaComputation computation)
      : computation_(std::move(computation)) {}

  lazy_tensors::StatusOr<lazy_tensors::ProgramShape> GetProgramShape()
      const override {
    xla::ProgramShape program_shape =
        ConsumeValue(computation_.GetProgramShape());
    return lazy_tensors::ProgramShape(
        torch_lazy_tensors::compiler::XlaHelpers::LazyTensorsShape(
            program_shape.result()),
        program_shape.parameters_size());
  }

  const XlaComputation& computation() const { return computation_; }

 private:
  XlaComputation computation_;
};

namespace compiler {

class NNCComputationClient : public lazy_tensors::ComputationClient {
 public:
  struct NNCData : public Data {
    NNCData(const at::Tensor& data, Shape shape, std::string device)
        : Data(std::move(device),
               torch_lazy_tensors::compiler::XlaHelpers::LazyTensorsShape(
                   std::move(shape))),
          data_(lazy_tensors::NNCComputationClient::HardwareDeviceType() ==
                        at::kCUDA
                    ? data.cuda()
                    : data) {}

    NNCData(Shape shape, std::string device)
        : Data(std::move(device),
               torch_lazy_tensors::compiler::XlaHelpers::LazyTensorsShape(
                   std::move(shape))) {}

    OpaqueHandle GetOpaqueHandle() override {
      return reinterpret_cast<int64>(this);
    }

    void Assign(const Data& data) override {
      data_ = static_cast<const NNCData&>(data).data_;
    }

    bool HasValue() const override { return data_.defined(); }

    at::Tensor data_;
  };

  DataPtr CreateDataPlaceholder(std::string device,
                                lazy_tensors::Shape shape) override;

  std::vector<DataPtr> TransferToServer(
      absl::Span<const TensorSource> tensors) override {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<lazy_tensors::Literal> TransferFromServer(
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

  std::map<std::string, lazy_tensors::Metric> GetMetrics() const override {
    return {};
  }

  MemoryInfo GetMemoryInfo(const std::string& device) override {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  void PrepareToExit() override;
};

lazy_tensors::ComputationClient* NNCGet();

lazy_tensors::ComputationClient* NNCGetIfInitialized();

}  // namespace compiler
}  // namespace xla
