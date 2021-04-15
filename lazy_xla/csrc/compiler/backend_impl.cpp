#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_xla/csrc/compiler/nnc_computation_client.h"
#include "lazy_xla/csrc/compiler/tensor_util.h"
#include "lazy_xla/csrc/compiler/xla_lowering_context.h"
#include "lazy_xla/csrc/compiler/xla_node_lowering.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"

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
      lazy_tensors::Span<const ir::Node* const> post_order,
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
      lazy_tensors::Span<const std::string> devices) const override {
    return std::vector<std::string>(devices.begin(), devices.end());
  }

  at::Tensor MakeTensorFromComputationData(
      const lazy_tensors::ComputationClient::DataPtr data,
      c10::optional<at::ScalarType> logical_scalar_type) const override {
    auto xla_literals =
        xla::ComputationClient::Get()->TransferFromServer({data});
    XLA_CHECK_EQ(xla_literals.size(), 1);
    XLA_CHECK(logical_scalar_type);
    return torch_lazy_tensors::xla_backend::MakeTensorFromXlaLiteral(
        xla_literals.front(), *logical_scalar_type);
  }

  lazy_tensors::ComputationClient::DataPtr MakeComputationDataFromTensor(
      const at::Tensor& tensor, const lazy_tensors::Shape& shape,
      const std::string& device) const override {
    std::vector<lazy_tensors::ComputationClient::TensorSource> source_tensors;
    Device physical_device(device);
    auto populate_fn =
        [&, device](
            const lazy_tensors::ComputationClient::TensorSource& source_tensor,
            void* dest_buffer, size_t dest_buffer_size) {
          PopulateTensorBuffer(tensor, source_tensor.shape, dest_buffer,
                               dest_buffer_size, physical_device);
        };
    source_tensors.emplace_back(lazy_tensors::ToShapeData(shape), device,
                                std::move(populate_fn));
    auto handles = lazy_tensors::ComputationClient::Get()->TransferToServer(
        source_tensors);
    LTC_CHECK_EQ(handles.size(), 1);
    return handles.front();
  }

  lazy_tensors::StatusOr<std::string> GetComputationBackendText(
      const lazy_tensors::GenericComputation* computation) const override {
    LTC_LOG(FATAL) << "Not implemented.";
  }
};

BackendImplInterface* GetXlaBackendImpl() {
  static XlaBackendImpl* xla_backend_impl = new XlaBackendImpl();
  return xla_backend_impl;
}

BackendRegistrar g_registrar(GetXlaBackendImpl());

}  // namespace compiler
}  // namespace torch_lazy_tensors
