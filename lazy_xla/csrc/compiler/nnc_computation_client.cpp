#include "lazy_xla/csrc/compiler/nnc_computation_client.h"

#include <chrono>
#include <future>

#include "absl/types/span.h"
#include "lazy_tensors/computation_client/multi_wait.h"
#include "lazy_tensors/computation_client/thread_pool.h"
#include "lazy_tensors/computation_client/unique.h"
#include "lazy_xla/csrc/compiler/debug_macros.h"
#include "lazy_xla/csrc/compiler/helpers.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
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

Shape ResultShape(const XlaComputation& xla_computation) {
  const auto maybe_program_shape = xla_computation.GetProgramShape();
  const auto& program_shape = maybe_program_shape.ValueOrDie();
  return program_shape.result();
}

// Launch each shard of the full loop on a different thread. If there's a single
// shard, just call it directly.
void LaunchComputation(const XlaComputation::CodeGen& codegen,
                       const std::vector<CodeGen::CallArg>& args) {
  if (codegen.codegen_shards.size() == 1) {
    codegen.codegen_shards.front()->call(args);
    return;
  }
  auto mwait = std::make_shared<lazy_tensors::util::MultiWait>(
      codegen.codegen_shards.size());
  for (const auto& codegen_shard : codegen.codegen_shards) {
    auto compute_fn = [codegen_shard, args]() { codegen_shard->call(args); };
    lazy_tensors::env::ScheduleClosure(
        lazy_tensors::util::MultiWait::Completer(mwait, std::move(compute_fn)));
  }
  mwait->Wait();
}

template <class T, at::ScalarType scalar_type>
lazy_tensors::ComputationClient::DataPtr ExecuteComputationImpl(
    const lazy_tensors::ComputationClient::Computation& computation,
    size_t output_idx, const Shape& result_shape,
    absl::Span<const lazy_tensors::ComputationClient::DataPtr> arguments,
    const std::string& device) {
  const auto& xla_computation =
      static_cast<const GenericComputationXla*>(computation.computation())
          ->computation();
  const auto& codegen = xla_computation.codegen(output_idx);
  if (codegen.parameter_number) {
    LTC_CHECK_LT(*codegen.parameter_number, arguments.size());
    return arguments[*codegen.parameter_number];
  }
  std::vector<int> result_sizes(result_shape.dimensions().begin(),
                                result_shape.dimensions().end());
  at::Tensor result_tensor;
  const auto it = codegen.output_to_input_aliases_.find(output_idx);
  if (it != codegen.output_to_input_aliases_.end()) {
    LTC_CHECK_LT(it->second, arguments.size());
    const auto nnc_data =
        std::static_pointer_cast<compiler::NNCComputationClient::NNCData>(
            arguments[it->second]);
    result_tensor = nnc_data->data_;
  } else {
    auto options =
        at::TensorOptions(scalar_type)
            .device(lazy_tensors::NNCComputationClient::HardwareDeviceType());
    result_tensor = at::empty(
        std::vector<int64_t>(result_sizes.begin(), result_sizes.end()),
        options);
  }
  std::vector<CodeGen::CallArg> args;
  lazy_tensors::util::Unique<at::Device> unique_device;
  std::vector<at::Tensor> args_contiguous;
  for (const auto& argument : arguments) {
    const auto nnc_data =
        std::static_pointer_cast<compiler::NNCComputationClient::NNCData>(
            argument);
    unique_device.set(nnc_data->data_.device());
    args_contiguous.push_back(nnc_data->data_.contiguous());
    args.emplace_back(args_contiguous.back().data_ptr());
  }
  unique_device.set(result_tensor.device());
  args.emplace_back(result_tensor.data_ptr());
  LaunchComputation(codegen, args);
  return std::make_shared<compiler::NNCComputationClient::NNCData>(
      result_tensor, result_shape, std::move(device));
}

bool ArgumentsOverlap(
    absl::Span<const lazy_tensors::ComputationClient::DataPtr> arguments) {
  if (arguments.empty()) {
    return false;
  }
  using Extent = std::pair<const char*, int64_t>;
  std::vector<Extent> extents;
  for (const auto& argument : arguments) {
    const auto nnc_data =
        std::static_pointer_cast<compiler::NNCComputationClient::NNCData>(
            argument);
    extents.emplace_back(
        static_cast<const char*>(nnc_data->data_.data_ptr()),
        nnc_data->data_.numel() * nnc_data->data_.element_size());
  }
  std::sort(
      extents.begin(), extents.end(),
      [](const Extent& e1, const Extent& e2) { return e1.first < e2.first; });
  LTC_CHECK(!extents.empty());
  const char* current_end = extents.front().first;
  for (const Extent& extent : extents) {
    if (extent.first < current_end) {
      return true;
    }
    current_end = extent.first + extent.second;
  }
  return false;
}

}  // namespace

namespace compiler {

lazy_tensors::ComputationClient::DataPtr
NNCComputationClient::CreateDataPlaceholder(std::string device,
                                            lazy_tensors::Shape shape) {
  return std::make_shared<NNCComputationClient::NNCData>(
      torch_lazy_tensors::compiler::XlaHelpers::XlaShape(std::move(shape)),
      std::move(device));
}

std::vector<lazy_tensors::ComputationClient::ComputationPtr>
NNCComputationClient::Compile(
    std::vector<ComputationClient::CompileInstance> instances) {
  std::vector<std::shared_ptr<ComputationClient::Computation>> computations;
  for (auto& instance : instances) {
    xla::ProgramShape program_shape = ConsumeValue(
        static_cast<const GenericComputationXla*>(instance.computation.get())
            ->computation()
            .GetProgramShape());
    lazy_tensors::ProgramShape lazy_program_shape(
        torch_lazy_tensors::compiler::XlaHelpers::LazyTensorsShape(
            program_shape.result()),
        program_shape.parameters_size());
    computations.push_back(
        std::make_shared<lazy_tensors::ComputationClient::Computation>(
            std::move(instance.computation), lazy_program_shape,
            instance.devices));
  }
  return computations;
}

std::vector<lazy_tensors::ComputationClient::DataPtr>
NNCComputationClient::ExecuteComputation(
    const lazy_tensors::ComputationClient::Computation& computation,
    absl::Span<const lazy_tensors::ComputationClient::DataPtr> arguments,
    const std::string& device,
    const lazy_tensors::ComputationClient::ExecuteComputationOptions& options) {
  const auto& xla_computation =
      static_cast<const GenericComputationXla*>(computation.computation())
          ->computation();
  auto result_shape = ResultShape(xla_computation);
  if (!result_shape.IsTuple()) {
    std::vector<Shape> component_shapes{result_shape};
    result_shape = Shape(absl::MakeSpan(component_shapes));
  }
  const size_t component_count = result_shape.tuple_shapes_size();
  auto mwait = std::make_shared<lazy_tensors::util::MultiWait>(component_count);
  std::vector<ComputationClient::DataPtr> result(component_count);
  // If argument storage overlaps, with input / output aliasing, launching
  // everything in parallel wouldn't be safe since different threads would write
  // to the same storage. This is overly conservative and we should still
  // parallelize the launch for non-overlapping groups.
  bool parallel = !ArgumentsOverlap(arguments) && component_count > 1;
  if (lazy_tensors::NNCComputationClient::HardwareDeviceType() == at::kCUDA) {
    parallel = false;
  }
  for (int idx = 0; idx < component_count; ++idx) {
    const auto& component_shape = result_shape.tuple_shapes(idx);
    std::function<void()> compute_fn;
    switch (component_shape.element_type()) {
      case PrimitiveType::S8: {
        compute_fn = [&computation, &result, &result_shape, idx, &arguments,
                      &device]() {
          const auto& component_shape = result_shape.tuple_shapes(idx);
          result[idx] = ExecuteComputationImpl<int8_t, at::kChar>(
              computation, idx, component_shape, arguments, device);
        };
        break;
      }
      case PrimitiveType::S16: {
        compute_fn = [&computation, &result, &result_shape, idx, &arguments,
                      &device]() {
          const auto& component_shape = result_shape.tuple_shapes(idx);
          result[idx] = ExecuteComputationImpl<int16_t, at::kShort>(
              computation, idx, component_shape, arguments, device);
        };
        break;
      }
      case PrimitiveType::S32: {
        compute_fn = [&computation, &result, &result_shape, idx, &arguments,
                      &device]() {
          const auto& component_shape = result_shape.tuple_shapes(idx);
          result[idx] = ExecuteComputationImpl<int32_t, at::kInt>(
              computation, idx, component_shape, arguments, device);
        };
        break;
      }
      case PrimitiveType::S64: {
        compute_fn = [&computation, &result, &result_shape, idx, &arguments,
                      &device]() {
          const auto& component_shape = result_shape.tuple_shapes(idx);
          result[idx] = ExecuteComputationImpl<int64_t, at::kLong>(
              computation, idx, component_shape, arguments, device);
        };
        break;
      }
      case PrimitiveType::U8: {
        compute_fn = [&computation, &result, &result_shape, idx, &arguments,
                      &device]() {
          const auto& component_shape = result_shape.tuple_shapes(idx);
          result[idx] = ExecuteComputationImpl<uint8_t, at::kByte>(
              computation, idx, component_shape, arguments, device);
        };
        break;
      }
      case PrimitiveType::F32: {
        compute_fn = [&computation, &result, &result_shape, idx, &arguments,
                      &device]() {
          const auto& component_shape = result_shape.tuple_shapes(idx);
          result[idx] = ExecuteComputationImpl<float, at::kFloat>(
              computation, idx, component_shape, arguments, device);
        };
        break;
      }
      case PrimitiveType::F64: {
        compute_fn = [&computation, &result, &result_shape, idx, &arguments,
                      &device]() {
          const auto& component_shape = result_shape.tuple_shapes(idx);
          result[idx] = ExecuteComputationImpl<double, at::kDouble>(
              computation, idx, component_shape, arguments, device);
        };
        break;
      }
      case PrimitiveType::PRED: {
        compute_fn = [&computation, &result, &result_shape, idx, &arguments,
                      &device]() {
          const auto& component_shape = result_shape.tuple_shapes(idx);
          result[idx] = ExecuteComputationImpl<uint8_t, at::kBool>(
              computation, idx, component_shape, arguments, device);
        };
        break;
      }
      default: {
        LOG(FATAL) << "Unsupported type: " << component_shape.element_type();
      }
    }
    if (parallel) {
      lazy_tensors::env::ScheduleClosure(
          lazy_tensors::util::MultiWait::Completer(mwait,
                                                   std::move(compute_fn)));
    } else {
      compute_fn();
    }
  }
  if (parallel) {
    mwait->Wait();
  }
  return result;
}

std::string NNCComputationClient::GetResourceDomain(
    const std::string& device) const {
  return "";
}

std::string NNCComputationClient::GetDefaultDevice() const {
  switch (lazy_tensors::NNCComputationClient::HardwareDeviceType()) {
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
