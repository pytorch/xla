#include "torch_xla/csrc/xla_backend_impl.h"

#include <ATen/ScalarOps.h>

#include "third_party/xla_client/debug_macros.h"
#include "torch_xla/csrc/computation.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
class XlaBackendImpl : public torch::lazy::BackendImplInterface {
 public:
  XlaBackendImpl() {}
  void PrepareToExit() const override { XLA_ERROR() << "Not implemented yet"; }

  void SetRngSeed(size_t seed) const override {
    XLA_ERROR() << "Not implemented yet";
  }

  const torch::lazy::IrBuilder* GetIrBuilder() const override {
    XLA_ERROR() << "Not implemented yet";
    return 0;
  }

  torch::lazy::BackendDataPtr MakeComputationDataFromTensor(
      const at::Tensor& tensor, const torch::lazy::Shape& shape,
      const torch::lazy::BackendDevice& device) const override {
    return TensorToXlaData(tensor, device);
  }

  torch::lazy::BackendDataPtr MakeComputationDataFromScalar(
      const at::Scalar& scalar,
      const torch::lazy::BackendDevice& device) const override {
    at::Tensor t = at::scalar_tensor(scalar);
    return TensorToXlaData(t, device);
  }

  torch::lazy::BackendDataPtr CreateDataPlaceholder(
      const torch::lazy::BackendDevice& device,
      const torch::lazy::Shape& shape) const override {
    xla::Shape xla_shape = MakeXlaShapeFromLazyShape(shape, device);
    return WrapXlaData(xla::ComputationClient::Get()->CreateDataPlaceholder(
        device.toString(), std::move(xla_shape)));
  }

  torch::lazy::BackendDataPtr GetComputationDataFromNode(
      torch::lazy::Node* node) const override {
    const DeviceData* device_data_node = DeviceData::Cast(node);
    if (!device_data_node) {
      return nullptr;
    }
    return device_data_node->data();
  }

  at::Tensor MakeTensorFromComputationData(
      const torch::lazy::BackendDataPtr data,
      c10::optional<at::ScalarType> logical_scalar_type) const override {
    // TODO(JackCaoG): handle the logical_scalar_type == nullptr case
    return XlaDataToTensors({data}, *logical_scalar_type)[0];
  }

  std::unique_ptr<torch::lazy::LoweringContext> CreateLoweringContext(
      const std::string& name, torch::lazy::BackendDevice device,
      c10::ArrayRef<torch::lazy::Node*> post_order,
      torch::lazy::Util::EmissionMap emit_status) const override {
    // TODO(JackCaoG): change LoweringContext to take post_order as
    // c10::ArrayRef<torch::lazy::Node*> instead of
    // c10::ArrayRef<const torch::lazy::Node*> since c10::ArrayRef already
    // provided const for its member.
    return std::make_unique<LoweringContext>(name, device);
  }

  std::unique_ptr<torch::lazy::LoweringContext> CreateLoweringContext(
      const std::string& name,
      torch::lazy::BackendDevice device) const override {
    return std::make_unique<LoweringContext>(name, device);
  }

  // GetCompilationDevices always returns devices today because
  // 1. devices passed to GetCompilationDevices is always empty
  // 2. ComputationClient's replication device is never set since we always have
  // one replica per process
  // We should revisit this API for SPMD
  std::vector<std::string> GetCompilationDevices(
      const std::string& device,
      c10::ArrayRef<std::string> devices) const override {
    return xla::ComputationClient::Get()->GetCompilationDevices(device,
                                                                devices);
  }

  std::vector<torch::lazy::ComputationPtr> Compile(
      std::vector<torch::lazy::ComputationPtr> instances) const override {
    std::vector<torch::lazy::ComputationPtr> res;
    std::vector<xla::ComputationClient::CompileInstance> compile_instances;
    torch::lazy::BackendDevice current_device = GetCurrentDevice();

    for (const torch::lazy::ComputationPtr instance : instances) {
      // TODO(JackCaoG): device is missing in instance, use CurrentDevice for
      // now
      const Computation* torch_xla_computation =
          dynamic_cast<Computation*>(instance.get());
      xla::Shape shape = MakeShapeWithDeviceLayout(
          torch_xla_computation->program_shape().result(),
          static_cast<XlaDeviceType>(current_device.type()));

      // Call GetCompilationDevices and passes all device here if needed.
      // Currently on TPU we always have 1 replica per device and one process
      // per device (except process 0 of each host which will have an additional
      // XLA:CPU), so compilation device is always the current device. This
      // won't be the case for SPMD. Note that after this call
      // torch_xla_computation->computation_ becomes invalid due to std::move.
      // TODO(JackCaoG): Verify this with GPU, we might only have 1 process with
      // multiple GPU as replica.
      compile_instances.push_back(xla::ComputationClient::CompileInstance(
          torch_xla_computation->move_computation(), current_device.toString(),
          {current_device.toString()}, &shape));
    }
    std::vector<std::shared_ptr<xla::ComputationClient::Computation>>
        client_computations = xla::ComputationClient::Get()->Compile(
            std::move(compile_instances));
    return WrapClientComputation(client_computations);
  }

  std::vector<torch::lazy::BackendDataPtr> ExecuteComputation(
      torch::lazy::Computation& computation,
      c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
      const torch::lazy::BackendDevice& device) const override {
    return {};
  }

  std::shared_ptr<torch::lazy::BackendDeviceType> GetDefaultDeviceType()
      const override {
    // want to reuse the getDefualtDeviceTypelogic
    torch::lazy::BackendDevice default_device = *GetDefaultDevice();
    return std::make_shared<DeviceType>(
        static_cast<XlaDeviceType>(default_device.type()));
  }

  at::DeviceType EagerFallbackDeviceType() const override {
    return at::DeviceType();
  }

  std::vector<torch::lazy::BackendDevice> GetBackendDevices() const override {
    return {};
  }

  torch::lazy::BackendDevice GetBackendDevice(
      c10::Device device) const override {
    return torch::lazy::BackendDevice();
  }

  std::string GetComputationBackendText(
      const torch::lazy::ComputationPtr computation) const override {
    return "";
  }
};

// torch::lazy::BackendImplInterface* GetXlaBackendImpl() {
//   static XlaBackendImpl* xla_backend_impl = new XlaBackendImpl();
//   return xla_backend_impl;
// }

void InitXlaBackend(){
    // TODO(JackCaoG): register backend
};

}  // namespace torch_xla