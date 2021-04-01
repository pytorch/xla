#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"
#include "lazy_xla/csrc/compiler/nnc_computation_client.h"
#include "lazy_xla/csrc/compiler/xla_lowering_context.h"
#include "lazy_xla/csrc/compiler/xla_node_lowering.h"

namespace torch_lazy_tensors {
namespace compiler {

class XlaBackendImpl : public BackendImplInterface {
 public:
  std::unique_ptr<NodeLowering> CreateNodeLowering(
      ir::LoweringContext* loctx) const override {
    return CreateXlaNodeLowering(loctx);
  }

  NodeLowering* GetNodeLowering() const override {
    return GetXlaNodeLowering();
  }

  std::unique_ptr<ir::LoweringContext> CreateLoweringContext(
      const std::string& name, Device device,
      absl::Span<const ir::Node* const> post_order,
      ir::Util::EmissionMap emit_status) const override {
    return std::make_unique<xla_backend::XlaLoweringContext>(
        name, device, post_order, emit_status);
  }

  std::unique_ptr<ir::LoweringContext> CreateLoweringContext(
      const std::string& name, Device device) const override {
    return std::make_unique<xla_backend::XlaLoweringContext>(name, device);
  }

  lazy_tensors::ComputationClient* GetComputationClient() const override {
    return xla::compiler::NNCGet();
  }

  lazy_tensors::ComputationClient* GetComputationClientIfInitialized()
      const override {
    return xla::compiler::NNCGetIfInitialized();
  }

  std::vector<std::string> GetCompilationDevices(
      const std::string& device,
      absl::Span<const std::string> devices) const override {
    return std::vector<std::string>(devices.begin(), devices.end());
  }

  at::Tensor MakeTensorFromComputationData(
      const lazy_tensors::ComputationClient::DataPtr data,
      c10::optional<at::ScalarType> logical_scalar_type) const override {
    const auto nnc_data =
        std::dynamic_pointer_cast<xla::compiler::NNCComputationClient::NNCData>(
            data);
    auto nnc_result = nnc_data->data_;
    if (logical_scalar_type &&
        nnc_result.scalar_type() != *logical_scalar_type) {
      nnc_result = nnc_result.to(*logical_scalar_type);
    }
    return nnc_result;
  }

  lazy_tensors::ComputationClient::DataPtr MakeComputationDataFromTensor(
      const at::Tensor& tensor, const lazy_tensors::Shape& shape,
      const std::string& device) const override {
    return std::make_shared<xla::compiler::NNCComputationClient::NNCData>(
        tensor, XlaHelpers::XlaShape(shape), device);
  }

  lazy_tensors::StatusOr<std::string> GetComputationBackendText(
      const lazy_tensors::GenericComputation* computation) const {
    LTC_LOG(FATAL) << "Not implemented.";
  }
};

BackendRegistrar g_registrar(new XlaBackendImpl());

}  // namespace compiler
}  // namespace torch_lazy_tensors
