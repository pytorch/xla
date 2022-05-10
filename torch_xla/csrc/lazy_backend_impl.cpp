#include "torch_xla/csrc/lazy_backend_impl.h"

#include "third_party/xla_client/debug_macros.h"

namespace torch_xla {

class XlaBackendImpl : public torch::lazy::BackendImplInterface {
  public:
    void PrepareToExit() override {
      // TODO @wonjoo implement in LTC phase 3
      XLA_ERROR() << "Not implemented yet";
    }

    void SetRngSeed(size_t seed) override {
      // TODO @wonjoo implement in LTC phase 3
      XLA_ERROR() << "Not implemented yet";
    }

    const IrBuilder* GetIrBuilder() override {
      // TODO @wonjoo implement in LTC phase 3
      static const IrBuilder* builder = new torch::lazy::IrBuilder();
      return builder;
    }

    BackendDataPtr MakeComputationDataFromTensor(
      const at::Tensor& tensor, const Shape& shape,
      const BackendDevice& device) override {

      }

    BackendDataPtr MakeComputationDataFromScalar(
      const at::Scalar& scalar,
      const torch::lazy::BackendDevice& device) override {

      }
  
    BackendDataPtr CreateDataPlaceholder(
      const BackendDevice& device, const Shape& shape) override {

      }

    BackendDataPtr GetComputationDataFromNode(Node*) override {

    }

    at::Tensor MakeTensorFromComputationData(
      const BackendDataPtr data,
      c10::optional<at::ScalarType> logical_scalar_type) override {

    }

    std::unique_ptr<LoweringContext> CreateLoweringContext(
      const std::string& name, BackendDevice device,
      c10::ArrayRef<torch::lazy::Node*> post_order,
      Util::EmissionMap emit_status) override {

    }

    std::unique_ptr<LoweringContext> CreateLoweringContext(
      const std::string& name, BackendDevice device) override {

    }

    std::vector<std::string> GetCompilationDevices(
      const std::string& device, c10::ArrayRef<std::string> devices) {

    }

    std::vector<ComputationPtr> Compile(
      std::vector<ComputationPtr> instances) override {

    }

    std::vector<BackendDataPtr> ExecuteComputation(
      Computation& computation, c10::ArrayRef<BackendDataPtr> arguments,
      const BackendDevice& device) override {

    }

    std::shared_ptr<BackendDeviceType> GetDefaultDeviceType() override {

    }

    at::DeviceType EagerFallbackDeviceType() override {

    }

    std::vector<BackendDevice> GetBackendDevices() override {

    }

    BackendDevice GetBackendDevice(c10::Device device) override {

    }

    std::string GetComputationBackendText(
      const ComputationPtr computation) override {

    }
}

torch::lazy::BackendImplInterface* GetXlaBackendImpl() {
  static XlaBackendImpl* xla_backend_impl = new XlaBackendImpl();
  return xla_backend_impl;
}

void InitXlaBackend() {

}

}  // namespace torch_xla