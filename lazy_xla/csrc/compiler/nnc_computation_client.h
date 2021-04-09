#pragma once

#include <ATen/core/Tensor.h>

#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/computation_client/nnc_computation_client.h"
#include "lazy_xla/csrc/compiler/helpers.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace xla {
namespace compiler {

class NNCComputationClient : public lazy_tensors::ComputationClient {
 public:
  DataPtr CreateDataPlaceholder(std::string device,
                                lazy_tensors::Shape shape) override;

  std::vector<DataPtr> TransferToServer(
      lazy_tensors::Span<const TensorSource> tensors) override;

  std::vector<lazy_tensors::Literal> TransferFromServer(
      lazy_tensors::Span<const DataPtr> handles) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) override;

  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation,
      lazy_tensors::Span<const DataPtr> arguments, const std::string& device,
      const ExecuteComputationOptions& options) override;

  std::vector<std::vector<DataPtr>> ExecuteReplicated(
      const Computation& computation,
      const std::vector<std::vector<DataPtr>>& arguments,
      lazy_tensors::Span<const std::string> devices,
      const ExecuteReplicatedOptions& options) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<std::vector<DataPtr>> ExecuteParallel(
      lazy_tensors::Span<const Computation* const> computations,
      const std::vector<std::vector<DataPtr>>& arguments,
      lazy_tensors::Span<const std::string> devices,
      const ExecuteParallelOptions& options) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<DataPtr> ExecuteChained(
      lazy_tensors::Span<const ExecuteChainedOp> ops,
      const std::string& device) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<std::vector<DataPtr>> DeconstructTuple(
      lazy_tensors::Span<const DataPtr> tuples) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
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
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::map<std::string, lazy_tensors::Metric> GetMetrics() const override {
    return {};
  }

  MemoryInfo GetMemoryInfo(const std::string& device) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  void PrepareToExit() override;
};

lazy_tensors::ComputationClient* NNCGet();

lazy_tensors::ComputationClient* NNCGetIfInitialized();

}  // namespace compiler
}  // namespace xla
