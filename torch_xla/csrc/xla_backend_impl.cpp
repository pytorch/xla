#include "torch_xla/csrc/xla_backend_impl.h"

#include <ATen/ScalarOps.h>

#include "third_party/xla_client/debug_macros.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/computation.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir_builder.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_util.h"

namespace at {
// This function is defined in the codegenerated RegisterDispatchKey.cpp file.
extern TORCH_API void RegisterXLAXLANativeFunctions();
extern TORCH_API void RegisterXLAAutogradXLANativeFunctions();
}  // namespace at

namespace torch_xla {
class XlaBackendImpl : public torch::lazy::BackendImplInterface {
 public:
  XlaBackendImpl() {}

  bool InitDefaultDeviceType() {
    if (!default_device_type_inited_) {
      // GetDefaultDevice will trigger the runtime device init, should
      // not do it during class init time.
      torch::lazy::BackendDevice default_device = *GetDefaultDevice();
      default_device_type_ = std::make_shared<DeviceType>(
          static_cast<XlaDeviceType>(default_device.type()));
      default_device_type_inited_ = true;
    }
    return true;
  }

  bool InitDefaultDeviceOrdinal() {
    if (!default_device_ordinal_inited_) {
      // GetDefaultDevice will trigger the runtime device init, should
      // not do it during class init time.
      torch::lazy::BackendDevice default_device = *GetDefaultDevice();
      default_device_ordinal_ = default_device.ordinal();
      default_device_ordinal_inited_ = true;
    }
    return true;
  }

  void PrepareToExit() const override { XLA_ERROR() << "Not implemented yet"; }

  void SetRngSeed(size_t seed) const override {
    // TODO(alanwaketan): This interface is not useful. We probably should
    // remove the base one.
    XLA_ERROR() << "Not implemented yet";
    return;
  }

  const torch::lazy::IrBuilder* GetIrBuilder() const override {
    static const torch::lazy::IrBuilder* builder = new XLAIrBuilder();
    return builder;
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
      const torch::lazy::Node* node) const override {
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
      c10::ArrayRef<const torch::lazy::Node*> post_order,
      torch::lazy::Util::EmissionMap emit_status) const override {
    return std::make_unique<LoweringContext>(name, device, post_order,
                                             emit_status);
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
    std::vector<xla::Shape> output_shapes;

    for (const torch::lazy::ComputationPtr instance : instances) {
      // TODO(JackCaoG): device is missing in instance, use CurrentDevice for
      // now
      const Computation* torch_xla_computation =
          dynamic_cast<Computation*>(instance.get());
      output_shapes.push_back(MakeShapeWithDeviceLayout(
          torch_xla_computation->program_shape().result(),
          static_cast<XlaDeviceType>(current_device.type())));

      // Call GetCompilationDevices and passes all device here if needed.
      // Currently on TPU we always have 1 replica per device and one process
      // per device (except process 0 of each host which will have an additional
      // XLA:CPU), so compilation device is always the current device. This
      // won't be the case for SPMD. Note that after this call
      // torch_xla_computation->computation_ becomes invalid due to std::move.
      // TODO(JackCaoG): Verify this with GPU, we might only have 1 process with
      // multiple GPU as replica.
      compile_instances.push_back(xla::ComputationClient::CompileInstance(
          torch_xla_computation->move_computation(),
          torch_xla_computation->get_device_string(),
          {current_device.toString()}, &output_shapes.back()));
    }
    std::vector<std::shared_ptr<xla::ComputationClient::Computation>>
        client_computations = xla::ComputationClient::Get()->Compile(
            std::move(compile_instances));
    return WrapClientComputation(client_computations);
  }

  std::vector<torch::lazy::BackendDataPtr> ExecuteComputation(
      torch::lazy::ComputationPtr computation,
      c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
      const torch::lazy::BackendDevice& device) const override {
    std::vector<xla::ComputationClient::DataPtr> results =
        xla::ComputationClient::Get()->ExecuteComputation(
            *(UnwrapClientComputation(computation).get()),
            UnwrapXlaData(arguments), device.toString());
    return WrapXlaData(results);
  }

  std::shared_ptr<torch::lazy::BackendDeviceType> GetDefaultDeviceType()
      const override {
    // lazily init default device type, we only need to init once.
    std::call_once(default_device_type_flag, [this] {
      const_cast<XlaBackendImpl*>(this)->InitDefaultDeviceType();
    });
    return default_device_type_;
  }

  void SetDefaultDeviceType(int8_t type) override {
    default_device_type_ =
        std::make_shared<DeviceType>(static_cast<XlaDeviceType>(type));
    default_device_type_inited_ = true;
  }

  int64_t GetDefaultDeviceOrdinal() const override {
    // lazily init default device ordinal, we only need to init once.
    std::call_once(default_device_ordinal_flag, [this] {
      const_cast<XlaBackendImpl*>(this)->InitDefaultDeviceOrdinal();
    });
    return default_device_ordinal_;
  }
  void SetDefaultDeviceOrdinal(int64_t ordinal) override {
    default_device_ordinal_ = ordinal;
    default_device_ordinal_inited_ = true;
  }

  at::DeviceType EagerFallbackDeviceType() const override {
    return at::DeviceType::CPU;
  }

  std::vector<torch::lazy::BackendDevice> GetBackendDevices() const override {
    return torch_xla::bridge::GetBackendDevices();
  }

  torch::lazy::BackendDevice GetBackendDevice(
      c10::Device device) const override {
    return torch_xla::bridge::AtenDeviceToXlaDevice(device);
  }

  std::string GetComputationBackendText(
      const torch::lazy::ComputationPtr computation) const override {
    return dynamic_cast<torch_xla::Computation*>(computation.get())
        ->to_string();
  }

 private:
  bool default_device_type_inited_ = false;
  bool default_device_ordinal_inited_ = false;
  std::shared_ptr<torch::lazy::BackendDeviceType> default_device_type_;
  int64_t default_device_ordinal_;
  mutable std::once_flag default_device_type_flag;
  mutable std::once_flag default_device_ordinal_flag;
};

torch::lazy::BackendImplInterface* GetXlaBackendImpl() {
  static XlaBackendImpl* xla_backend_impl = new XlaBackendImpl();
  return xla_backend_impl;
}

torch::lazy::LazyGraphExecutor* GetXlaLazyGraphExecutor() {
  static torch::lazy::LazyGraphExecutor* executor =
      new torch::lazy::LazyGraphExecutor();
  return executor;
}

bool InitXlaBackend() {
  static std::once_flag register_key_flag;
  // Registration should only happen once.
  std::call_once(register_key_flag, [] {
    TORCH_LAZY_COUNTER("RegisterXLAFunctions", 1);
    // xla_fallback is currently auto registered when initializing torch_xla. No
    // need to re-register here.
    at::RegisterXLAXLANativeFunctions();
    at::RegisterXLAAutogradXLANativeFunctions();
  });
  static std::unique_ptr<torch::lazy::BackendRegistrar> s_registrar;
  s_registrar =
      std::make_unique<torch::lazy::BackendRegistrar>(GetXlaBackendImpl());
  torch::lazy::LazyGraphExecutor::Register(GetXlaLazyGraphExecutor());
  return true;
};

}  // namespace torch_xla
