#include "lazy_xla/csrc/compiler/nnc_computation_client.h"

#include <chrono>
#include <future>

#include "absl/types/span.h"
#include "lazy_tensors/computation_client/multi_wait.h"
#include "lazy_tensors/computation_client/thread_pool.h"
#include "lazy_tensors/computation_client/unique.h"
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

}  // namespace

namespace compiler {

lazy_tensors::ComputationClient::DataPtr
NNCComputationClient::CreateDataPlaceholder(std::string device,
                                            lazy_tensors::Shape shape) {
  LTC_LOG(FATAL) << "Not supported.";
}

std::vector<lazy_tensors::ComputationClient::ComputationPtr>
NNCComputationClient::Compile(
    std::vector<ComputationClient::CompileInstance> instances) {
  LTC_LOG(FATAL) << "Not supported.";
}

std::vector<lazy_tensors::ComputationClient::DataPtr>
NNCComputationClient::ExecuteComputation(
    const lazy_tensors::ComputationClient::Computation& computation,
    lazy_tensors::Span<const lazy_tensors::ComputationClient::DataPtr>
        arguments,
    const std::string& device,
    const lazy_tensors::ComputationClient::ExecuteComputationOptions& options) {
  LTC_LOG(FATAL) << "Not supported.";
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
