#include "lazy_xla/csrc/compiler/nnc_computation_client.h"

#include <chrono>
#include <future>

#include "absl/types/span.h"
#include "lazy_tensors/computation_client/multi_wait.h"
#include "lazy_tensors/computation_client/thread_pool.h"
#include "lazy_tensors/computation_client/unique.h"
#include "lazy_xla/csrc/compiler/helpers.h"
#include "lazy_xla/csrc/compiler/xla_lowering_context.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "test/cpp/tensorexpr/padded_buffer.h"
#include "torch/csrc/jit/tensorexpr/ir_simplifier.h"

using namespace torch::jit::tensorexpr;

namespace xla {
namespace {

std::atomic<lazy_tensors::ComputationClient*> g_computation_client(nullptr);
std::once_flag g_computation_client_once;

lazy_tensors::ComputationClient* CreateClient() {
  return new compiler::NNCComputationClient();
}

struct XrtComputation : lazy_tensors::ComputationClient::Computation {
 public:
  XrtComputation(
      std::shared_ptr<lazy_tensors::GenericComputation> computation,
      lazy_tensors::ProgramShape program_shape,
      std::vector<std::string> devices,
      std::shared_ptr<xla::XrtComputationClient::XrtComputation> original)
      : lazy_tensors::ComputationClient::Computation(computation, program_shape,
                                                     devices),
        original(original) {}

  std::shared_ptr<xla::XrtComputationClient::XrtComputation> original;
};

}  // namespace

namespace compiler {

lazy_tensors::ComputationClient::DataPtr
NNCComputationClient::CreateDataPlaceholder(std::string device,
                                            lazy_tensors::Shape shape) {
  return xla::ComputationClient::Get()->CreateDataPlaceholder(
      device, lazy_tensors::ToShapeData(shape));
}

std::vector<lazy_tensors::ComputationClient::DataPtr>
NNCComputationClient::TransferToServer(
    lazy_tensors::Span<const TensorSource> tensors) {
  return xla::ComputationClient::Get()->TransferToServer(tensors);
}

std::vector<lazy_tensors::ComputationClient::ComputationPtr>
NNCComputationClient::Compile(
    std::vector<ComputationClient::CompileInstance> instances) {
  std::list<Shape> xla_shapes;
  std::vector<xla::ComputationClient::CompileInstance> xla_instances;
  for (auto& instance : instances) {
    xla::XlaComputation xla_computation =
        std::static_pointer_cast<
            torch_lazy_tensors::compiler::xla_backend::GenericComputationXla>(
            instance.computation)
            ->move_computation();
    Shape* xla_shape = nullptr;
    if (instance.output_shape) {
      xla_shapes.push_back(torch_lazy_tensors::compiler::XlaHelpers::XlaShape(
          *instance.output_shape));
      xla_shape = &xla_shapes.back();
    }
    xla_instances.emplace_back(std::move(xla_computation),
                               instance.compilation_device, instance.devices,
                               xla_shape);
  }
  std::vector<lazy_tensors::ComputationClient::ComputationPtr> result;
  std::vector<std::string> compilation_devices;
  for (auto& instance : instances) {
    compilation_devices.push_back(instance.compilation_device);
  }
  auto xla_result =
      xla::ComputationClient::Get()->Compile(std::move(xla_instances));
  XLA_CHECK_EQ(xla_result.size(), compilation_devices.size());
  for (size_t i = 0; i < xla_result.size(); ++i) {
    auto xrt_computation =
        std::static_pointer_cast<xla::XrtComputationClient::XrtComputation>(
            xla_result[i]);
    auto generic_computation = std::make_shared<
        torch_lazy_tensors::compiler::xla_backend::GenericComputationXla>(
        xrt_computation->move_computation());
    const xla::ProgramShape& xla_program_shape =
        xrt_computation->program_shape();
    std::vector<lazy_tensors::Shape> parameter_shapes;
    parameter_shapes.reserve(xla_program_shape.parameters_size());
    for (const xla::Shape& xla_parameter_shape :
         xla_program_shape.parameters()) {
      parameter_shapes.push_back(
          torch_lazy_tensors::compiler::XlaHelpers::LazyTensorsShape(
              xla_parameter_shape));
    }
    lazy_tensors::Shape result_shape =
        torch_lazy_tensors::compiler::XlaHelpers::LazyTensorsShape(
            xla_program_shape.result());
    lazy_tensors::ProgramShape program_shape(
        parameter_shapes, xla_program_shape.parameter_names(), result_shape);
    result.push_back(std::make_shared<XrtComputation>(
        generic_computation, program_shape, xrt_computation->devices(),
        xrt_computation));
  }
  return result;
}

std::vector<lazy_tensors::ComputationClient::DataPtr>
NNCComputationClient::ExecuteComputation(
    const lazy_tensors::ComputationClient::Computation& computation,
    lazy_tensors::Span<const lazy_tensors::ComputationClient::DataPtr>
        arguments,
    const std::string& device,
    const lazy_tensors::ComputationClient::ExecuteComputationOptions& options) {
  const XrtComputation& xrt_computation =
      dynamic_cast<const XrtComputation&>(computation);
  xla::XlaComputation xla_computation =
      static_cast<
          torch_lazy_tensors::compiler::xla_backend::GenericComputationXla*>(
          computation.computation())
          ->move_computation();
  xla::ProgramShape program_shape;
  for (const lazy_tensors::Shape& parameter_shape :
       computation.program_shape().parameters()) {
    *program_shape.add_parameters() =
        torch_lazy_tensors::compiler::XlaHelpers::XlaShape(parameter_shape);
  }
  *program_shape.mutable_result() =
      torch_lazy_tensors::compiler::XlaHelpers::XlaShape(
          computation.program_shape().result());
  xla::ComputationClient::ExecuteComputationOptions xla_options;
  xla_options.explode_tuple = options.explode_tuple;
  return xla::ComputationClient::Get()->ExecuteComputation(
      *xrt_computation.original, arguments, device, xla_options);
}

std::string NNCComputationClient::GetResourceDomain(
    const std::string& device) const {
  return "";
}

std::string NNCComputationClient::GetDefaultDevice() const {
  switch (HardwareDeviceType()) {
    case at::kCPU: {
      return "CPU:0";
    }
    case at::kCUDA: {
      return "GPU:0";
    }
    default: { LTC_LOG(FATAL) << "Invalid device type"; }
  }
}

std::vector<std::string> NNCComputationClient::GetLocalDevices() const {
  return {GetDefaultDevice()};
}

std::vector<std::string> NNCComputationClient::GetAllDevices() const {
  return GetLocalDevices();
}

void NNCComputationClient::SetReplicationDevices(
    std::shared_ptr<std::vector<std::string>> devices) {
  LTC_CHECK_EQ(devices->size(), size_t(1)) << "Replication not supported yet";
}

std::shared_ptr<std::vector<std::string>>
NNCComputationClient::GetReplicationDevices() {
  return nullptr;
}

void NNCComputationClient::PrepareToExit() {}

at::DeviceType NNCComputationClient::HardwareDeviceType() {
  static auto device_type =
      sys_util::GetEnvBool("NNC_CUDA", false) ? at::kCUDA : at::kCPU;
  // The first CUDA usage could happen via lazy tensors. Initialize CUDA here to
  // account for that, at::scalar_tensor constructor triggers everything we
  // need.
  static c10::optional<at::Tensor> init_cuda =
      device_type == at::kCUDA ? c10::optional<at::Tensor>(at::scalar_tensor(
                                     0, at::TensorOptions().device(at::kCUDA)))
                               : c10::nullopt;
  return device_type;
}

lazy_tensors::ComputationClient* NNCGet() {
  std::call_once(g_computation_client_once,
                 [&]() { g_computation_client = CreateClient(); });
  return g_computation_client.load();
}

lazy_tensors::ComputationClient* NNCGetIfInitialized() {
  return g_computation_client.load();
}

}  // namespace compiler
}  // namespace xla
