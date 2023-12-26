#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/profiler.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch_xla/csrc/runtime/xla_coordinator.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"

namespace torch_xla {
namespace runtime {

std::unordered_map<std::string, std::string> pjrt_plugins_;

namespace {

xla::GpuAllocatorConfig GetGpuAllocatorConfig() {
  auto allocator_config = xla::GpuAllocatorConfig{};
  if (sys_util::GetEnvString(env::kEnvPjrtAllocatorCudaAsync, "").empty() &&
      sys_util::GetEnvString(env::kEnvPjrtAllocatorPreallocate, "").empty() &&
      sys_util::GetEnvString(env::kEnvPjrtAllocatorFraction, "").empty()) {
    return allocator_config;
  }
  if (sys_util::GetEnvBool(env::kEnvPjrtAllocatorCudaAsync, false)) {
    allocator_config.kind = xla::GpuAllocatorConfig::Kind::kCudaAsync;
  }
  allocator_config.preallocate =
      sys_util::GetEnvBool(env::kEnvPjrtAllocatorPreallocate, true);
  allocator_config.memory_fraction =
      sys_util::GetEnvDouble(env::kEnvPjrtAllocatorFraction, 0.75);
  return allocator_config;
}

std::optional<std::string> GetPjRtPluginPath(const std::string& device_type) {
  auto plugin_path = pjrt_plugins_.find(device_type);
  return plugin_path != pjrt_plugins_.end() ? std::optional(plugin_path->second)
                                            : std::nullopt;
}

}  // namespace

void RegisterPjRtPlugin(std::string name, std::string library_path) {
  TF_VLOG(3) << "Registering PjRt plugin " << name << " at " << library_path;
  pjrt_plugins_[name] = library_path;
}

std::tuple<std::unique_ptr<xla::PjRtClient>, std::unique_ptr<XlaCoordinator>>
InitializePjRt(const std::string& device_type) {
  std::unique_ptr<xla::PjRtClient> client;
  std::unique_ptr<XlaCoordinator> coordinator;

  if (sys_util::GetEnvBool(env::kEnvPjrtDynamicPlugins, false)) {
    std::optional<std::string> plugin_path = GetPjRtPluginPath(device_type);
    if (plugin_path) {
      TF_VLOG(1) << "Initializing client for PjRt plugin " << device_type;
      const PJRT_Api* c_api = *pjrt::LoadPjrtPlugin(
          absl::AsciiStrToLower(device_type), *plugin_path);
      XLA_CHECK_OK(pjrt::InitializePjrtPlugin(device_type));
      client = xla::GetCApiClient(absl::AsciiStrToUpper(device_type)).value();
      profiler::RegisterProfilerForPlugin(c_api);
    }
  } else if (device_type == "CPU") {
    TF_VLOG(1) << "Initializing PjRt CPU client...";
    bool async = sys_util::GetEnvBool(env::kEnvPjrtAsyncCpuClient, true);
    int cpu_device_count = sys_util::GetEnvInt(env::kEnvNumCpu, 1);
    client = std::move(xla::GetTfrtCpuClient(async, cpu_device_count).value());
  } else if (device_type == "TPU" || device_type == "TPU_C_API") {
    TF_VLOG(1) << "Initializing TFRT TPU client...";
    // Prefer $TPU_LIBRARY_PATH if set
    auto tpu_library_path = sys_util::GetEnvString(
        env::kEnvTpuLibraryPath,
        sys_util::GetEnvString(env::kEnvInferredTpuLibraryPath, "libtpu.so"));
    XLA_CHECK_OK(pjrt::LoadPjrtPlugin("tpu", tpu_library_path).status());
    tsl::Status tpu_status = pjrt::InitializePjrtPlugin("tpu");
    XLA_CHECK_OK(tpu_status);
    client = std::move(xla::GetCApiClient("TPU").value());
    const PJRT_Api* c_api =
        static_cast<xla::PjRtCApiClient*>(client.get())->pjrt_c_api();
    profiler::RegisterProfilerForPlugin(c_api);
  } else if (device_type == "TPU_LEGACY") {
    XLA_ERROR() << "TPU_LEGACY client is no longer available.";
  } else if (device_type == "CUDA") {
    TF_VLOG(1) << "Initializing PjRt GPU client...";
    bool async = sys_util::GetEnvBool(env::kEnvPjrtAsyncGpuClient, true);
    int local_process_rank = sys_util::GetEnvInt(env::kEnvPjRtLocalRank, 0);
    int global_process_rank = sys_util::GetEnvInt("RANK", local_process_rank);
    int local_world_size = sys_util::GetEnvInt("LOCAL_WORLD_SIZE", 1);
    int global_world_size = sys_util::GetEnvInt("WORLD_SIZE", local_world_size);
    std::string master_addr =
        runtime::sys_util::GetEnvString("MASTER_ADDR", "localhost");
    std::string port = runtime::sys_util::GetEnvString(
        "XLA_COORDINATOR_PORT", XlaCoordinator::kDefaultCoordinatorPort);

    xla::PjRtClient::KeyValueGetCallback kv_get = nullptr;
    xla::PjRtClient::KeyValuePutCallback kv_put = nullptr;
    auto allowed_devices =
        std::make_optional<std::set<int>>(std::set{local_process_rank});
    if (global_world_size > 1) {
      // Use the XlaCoordinator as the distributed key-value store.
      coordinator = std::make_unique<XlaCoordinator>(
          global_process_rank, global_world_size, master_addr, port);
      std::shared_ptr<xla::DistributedRuntimeClient> distributed_client =
          coordinator->GetClient();
      std::string key_prefix = "gpu:";
      kv_get = [distributed_client, key_prefix](
                   std::string_view k,
                   absl::Duration timeout) -> xla::StatusOr<std::string> {
        return distributed_client->BlockingKeyValueGet(
            absl::StrCat(key_prefix, k), timeout);
      };
      kv_put = [distributed_client, key_prefix](
                   std::string_view k, std::string_view v) -> xla::Status {
        return distributed_client->KeyValueSet(absl::StrCat(key_prefix, k), v);
      };
    }
    TF_VLOG(3) << "Getting StreamExecutorGpuClient for node_id="
               << global_process_rank << ", num_nodes=" << global_world_size;
    xla::GpuClientOptions options;
    options.allocator_config = GetGpuAllocatorConfig();
    options.node_id = global_process_rank;
    options.num_nodes = global_world_size;
    options.allowed_devices = allowed_devices;
    options.platform_name = "gpu";
    options.should_stage_host_to_device_transfers = true;
    options.kv_get = kv_get;
    options.kv_put = kv_put;
    client = std::move(xla::GetStreamExecutorGpuClient(options).value());
  } else if (device_type == "XPU") {
    TF_VLOG(1) << "Initializing PjRt XPU client...";
    XLA_CHECK_OK(
        pjrt::LoadPjrtPlugin(
            "xpu", sys_util::GetEnvString(env::kEnvXpuLibraryPath, "libxpu.so"))
            .status());
    client = std::move(xla::GetCApiClient("XPU").value());
  } else if (device_type == "NEURON") {
    TF_VLOG(1) << "Initializing PjRt NEURON client...";
    XLA_CHECK_OK(pjrt::LoadPjrtPlugin("NEURON", sys_util::GetEnvString(
                                                    env::kEnvNeuronLibraryPath,
                                                    "libneuronpjrt.so"))
                     .status());
    client = std::move(xla::GetCApiClient("NEURON").value());
  }

  XLA_CHECK(client) << absl::StrFormat("Unknown %s '%s'", env::kEnvPjRtDevice,
                                       device_type);

  return {std::move(client), std::move(coordinator)};
}

}  // namespace runtime
}  // namespace torch_xla
