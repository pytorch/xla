#include "third_party/xla_client/computation_client.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/platform/net.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/env_vars.h"
#include "third_party/xla_client/mesh_service.h"
#include "third_party/xla_client/pjrt_computation_client.h"
#include "third_party/xla_client/sys_util.h"
#include "third_party/xla_client/xrt_computation_client.h"

namespace xla {
namespace {

struct DeviceCountDefaults {
  int num_tpus = 0;
  int num_gpus = 0;
  int num_cpus = 1;
};

std::atomic<ComputationClient*> g_computation_client(nullptr);
std::once_flag g_computation_client_once;

ComputationClient* CreateClient() {
  if (sys_util::GetEnvBool("XLA_DUMP_FATAL_STACK", false)) {
    tensorflow::testing::InstallStacktraceHandler();
  }
  auto client = ComputationClient::Create();
  return client.release();
}

std::string MakeGrpcEndPoint(const std::string& server) {
  return server.compare(0, 7, "grpc://") == 0 ? server
                                              : absl::StrCat("grpc://", server);
}

std::string GetXrtDevicePath(const std::string& worker, int task_no,
                             const std::string& device_type, int ordinal) {
  return absl::StrCat("/job:", worker, "/replica:0/task:", task_no,
                      "/device:", device_type, ":", ordinal);
}

std::string BuildTaskDeviceKey(int task_no, const std::string& kind) {
  return absl::StrCat(task_no, ":", kind);
}

tensorflow::DeviceNameUtils::ParsedName ParseXrtDevice(
    const std::string& device) {
  tensorflow::DeviceNameUtils::ParsedName parsed_device;
  XLA_CHECK(
      tensorflow::DeviceNameUtils::ParseFullName(device, &parsed_device) &&
      parsed_device.has_job && parsed_device.has_task && parsed_device.has_id &&
      parsed_device.has_type)
      << device;
  return parsed_device;
}

bool IsLocalDevice(const XrtComputationClient::Worker& worker,
                   const tensorflow::DeviceNameUtils::ParsedName& parsed_device,
                   const std::map<std::string, int>& dev_task_map) {
  if (worker.name != parsed_device.job ||
      worker.task_no != parsed_device.task) {
    return false;
  }
  std::string mp_device = XrtComputationClient::GetMultiProcessingDevice();
  if (mp_device.empty()) {
    return true;
  }
  XrtComputationClient::Device device(mp_device);
  std::string task_device_key =
      BuildTaskDeviceKey(parsed_device.task, device.kind);
  auto it = dev_task_map.find(task_device_key);
  return it != dev_task_map.end()
             ? (device.ordinal == it->second + parsed_device.id)
             : false;
}

std::map<std::string, int> BuildDeviceTaskMap(
    const XrtComputationClient::Options& options) {
  // Builds a map from "TASK:DEV_KIND" (ie, "0:TPU") keys to the minimum global
  // device ordinal assigned for that task+devkind couple.
  std::map<std::string, int> dev_task_map;
  for (auto& device_xrt_device : options.global_device_map) {
    XrtComputationClient::Device global_device(device_xrt_device.first);
    tensorflow::DeviceNameUtils::ParsedName parsed_device =
        ParseXrtDevice(device_xrt_device.second);
    std::string task_device_key =
        BuildTaskDeviceKey(parsed_device.task, global_device.kind);
    util::InsertCombined(&dev_task_map, task_device_key, global_device.ordinal,
                         [](int a, int b) { return std::min(a, b); });
  }
  return dev_task_map;
}

void PopulateLocalDevices(XrtComputationClient::Options* options) {
  std::string local_worker = sys_util::GetEnvString(env::kEnvLocalWorker, "");
  XrtComputationClient::Worker worker("", -1);
  if (!local_worker.empty()) {
    worker = XrtComputationClient::ParseWorker(local_worker);
  }
  auto dev_task_map = BuildDeviceTaskMap(*options);
  std::map<std::string, int> min_ordinals;
  for (auto& device_xrt_device : options->global_device_map) {
    if (worker.task_no >= 0) {
      tensorflow::DeviceNameUtils::ParsedName parsed_device =
          ParseXrtDevice(device_xrt_device.second);
      if (!IsLocalDevice(worker, parsed_device, dev_task_map)) {
        continue;
      }
    }
    options->devices.insert(device_xrt_device.first);

    XrtComputationClient::Device global_device(device_xrt_device.first);
    util::InsertCombined(&min_ordinals, global_device.kind,
                         global_device.ordinal,
                         [](int a, int b) { return std::min(a, b); });
  }
  for (auto kind : {"TPU", "GPU", "CPU"}) {
    auto it = min_ordinals.find(kind);
    if (it != min_ordinals.end()) {
      options->default_device = absl::StrCat(kind, ":", it->second);
      break;
    }
  }
}

void AddXrtHostDevices(const std::string& worker_name, int task_no,
                       const std::string& server,
                       const DeviceCountDefaults& device_counts,
                       std::map<std::string, int>* device_ordinals,
                       XrtComputationClient::Options* options) {
  struct Devices {
    const char* name;
    const char* tf_name;
    int64_t count;
  } const devices[] = {
      {"TPU", "TPU",
       sys_util::GetEnvInt(env::kEnvNumTpu, device_counts.num_tpus)},
      {"GPU", "XLA_GPU",
       sys_util::GetEnvInt(env::kEnvNumGpu, device_counts.num_gpus)},
      {"CPU", "XLA_CPU", device_counts.num_cpus},
  };
  options->workers_map.emplace(
      XrtComputationClient::Worker(worker_name, task_no),
      MakeGrpcEndPoint(server));
  for (auto& device : devices) {
    int& device_ordinal = (*device_ordinals)[device.name];
    for (int j = 0; j < device.count; ++j, ++device_ordinal) {
      std::string device_name = absl::StrCat(device.name, ":", device_ordinal);
      std::string xrt_device_name =
          GetXrtDevicePath(worker_name, task_no, device.tf_name, j);
      options->global_device_map.emplace(device_name, xrt_device_name);
    }
  }
}

bool ParseEnvBasedTpuClusterConfig(XrtComputationClient::Options* options) {
  std::string tpu_config = sys_util::GetEnvString(env::kEnvTpuConfig, "");
  if (tpu_config.empty()) {
    return false;
  }
  std::map<std::string, int> device_ordinals;
  std::vector<std::string> spec_parts = absl::StrSplit(tpu_config, '|');
  XLA_CHECK(!spec_parts.empty()) << tpu_config;
  DeviceCountDefaults device_counts;
  device_counts.num_tpus = 8;
  for (const auto& spec : spec_parts) {
    std::vector<std::string> host_parts = absl::StrSplit(spec, ';');
    XLA_CHECK_EQ(host_parts.size(), 3) << spec;
    AddXrtHostDevices(host_parts[0], std::stoi(host_parts[1]), host_parts[2],
                      device_counts, &device_ordinals, options);
  }
  return true;
}

bool ParseMeshConfig(
    XrtComputationClient::Options* options,
    std::unique_ptr<tensorflow::tpu::TopologyProto>* topology_proto) {
  service::MeshClient* client = service::MeshClient::Get();
  if (client == nullptr) {
    return false;
  }
  std::string local_worker_env =
      sys_util::GetEnvString(env::kEnvLocalWorker, "");
  XLA_CHECK(!local_worker_env.empty())
      << "In a mesh client setup the XRT_LOCAL_WORKER must be specified";

  XrtComputationClient::Worker local_worker =
      XrtComputationClient::ParseWorker(local_worker_env);
  int host_ordinal = sys_util::GetEnvInt(env::kEnvHostOrdinal, 0);

  TF_LOG(INFO) << "Fetching mesh configuration for worker " << local_worker.name
               << " (host_ordinal=" << host_ordinal
               << "):" << local_worker.task_no << " from mesh service at "
               << client->address();
  service::grpc::Config config = client->GetConfig(host_ordinal);
  TF_VLOG(3) << "Mesh Config: " << config.DebugString();

  std::string mp_device = XrtComputationClient::GetMultiProcessingDevice();
  for (auto& config_worker : config.workers()) {
    XrtComputationClient::Worker worker(config_worker.name(),
                                        config_worker.task_no());
    options->workers_map.emplace(worker, config_worker.address());

    for (auto& device : config_worker.devices()) {
      XrtComputationClient::Device local_device(device.local_name());
      options->global_device_map.emplace(
          device.global_name(),
          GetXrtDevicePath(worker.name, worker.task_no, local_device.kind,
                           local_device.ordinal));
      if (local_worker == worker &&
          (mp_device.empty() || device.global_name() == mp_device)) {
        options->devices.insert(device.global_name());
      }
    }
  }
  (*topology_proto) = absl::make_unique<tensorflow::tpu::TopologyProto>(
      std::move(*config.mutable_proto()));
  return true;
}

bool ParseEnvDeviceCounts(XrtComputationClient::Options* options) {
  DeviceCountDefaults device_counts;
  device_counts.num_tpus = sys_util::GetEnvInt(env::kEnvNumTpu, 0);
  device_counts.num_gpus = sys_util::GetEnvInt(env::kEnvNumGpu, 0);
  if (device_counts.num_tpus > 0 || device_counts.num_gpus > 0) {
    std::map<std::string, int> device_ordinals;
    std::string host_port =
        absl::StrCat("localhost:", tensorflow::internal::PickUnusedPortOrDie());
    AddXrtHostDevices("localservice", 0, host_port, device_counts,
                      &device_ordinals, options);
  }
  return !options->global_device_map.empty();
}

bool ParseEnvDevices(XrtComputationClient::Options* options) {
  std::string device_spec = sys_util::GetEnvString(env::kEnvDeviceMap, "");
  std::string workers_spec = sys_util::GetEnvString(env::kEnvWorkers, "");
  if (!device_spec.empty() && !workers_spec.empty()) {
    for (const auto& device_target : absl::StrSplit(device_spec, '|')) {
      std::vector<std::string> parts = absl::StrSplit(device_target, ';');
      XLA_CHECK_EQ(parts.size(), 2) << device_target;
      options->global_device_map.emplace(parts[0], parts[1]);
    }
    for (const auto& name_target : absl::StrSplit(workers_spec, '|')) {
      std::vector<std::string> parts = absl::StrSplit(name_target, ';');
      XLA_CHECK_EQ(parts.size(), 2) << name_target;
      options->workers_map.emplace(XrtComputationClient::ParseWorker(parts[0]),
                                   MakeGrpcEndPoint(parts[1]));
    }
  }
  return !options->global_device_map.empty();
}
}  // namespace

std::unique_ptr<ComputationClient> ComputationClient::Create() {
  std::unique_ptr<ComputationClient> client;

  if (sys_util::GetEnvString(env::kEnvPjRtDevice, "") != "") {
    client = std::unique_ptr<ComputationClient>(new PjRtComputationClient());
  } else {
    XrtComputationClient::Options options;
    std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto;
    if (!ParseEnvBasedTpuClusterConfig(&options) &&
        !ParseEnvDeviceCounts(&options) && !ParseEnvDevices(&options) &&
        !ParseMeshConfig(&options, &topology_proto)) {
      XLA_ERROR() << "Missing XLA configuration";
    }
    PopulateLocalDevices(&options);
    client = std::unique_ptr<ComputationClient>(
        new XrtComputationClient(options, std::move(topology_proto)));
  }

  XLA_CHECK(client.get() != nullptr);
  return client;
}

std::shared_ptr<ComputationClient::Computation> ComputationClient::Compile(
    XlaComputation computation, std::string compilation_device,
    std::vector<std::string> devices, const Shape* output_shape) {
  std::vector<CompileInstance> instances;
  instances.emplace_back(std::move(computation), std::move(compilation_device),
                         std::move(devices), output_shape);
  std::vector<std::shared_ptr<Computation>> results =
      Compile(std::move(instances));
  return std::move(results[0]);
}

std::vector<std::string> ComputationClient::GetCompilationDevices(
    const std::string& device, absl::Span<const std::string> devices) {
  std::vector<std::string> compilation_devices;
  if (devices.empty()) {
    auto replication_devices = GetReplicationDevices();
    if (replication_devices == nullptr || replication_devices->empty()) {
      compilation_devices.push_back(device);
    } else {
      compilation_devices = *replication_devices;
    }
  } else {
    compilation_devices.insert(compilation_devices.end(), devices.begin(),
                               devices.end());
  }
  return compilation_devices;
}

void ComputationClient::RunLocalService(uint64_t service_port) {
  try {
    XrtLocalService* service = new XrtLocalService(
        "localservice|localhost:" + std::to_string(service_port),
        "localservice", 0);
    service->Start();
    service->Join();
  } catch (const std::runtime_error& error) {
    if (std::string(error.what()).find("Couldn't open device: /dev/accel0") !=
        std::string::npos) {
      TF_LOG(INFO) << "Local service has been created by other process, return";
    } else {
      throw;
    }
  }
}

int64_t ComputationClient::GetDeviceOrdinal(const std::string& device) {
  auto pos = device.rfind(':');
  XLA_CHECK_NE(pos, std::string::npos) << device;
  return std::stoi(device.substr(pos + 1));
}

ComputationClient* ComputationClient::Get() {
  std::call_once(g_computation_client_once,
                 [&]() { g_computation_client = CreateClient(); });
  return g_computation_client.load();
}

ComputationClient* ComputationClient::GetIfInitialized() {
  return g_computation_client.load();
}

metrics::Metric* ComputationClient::TransferToServerMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("TransferToServerTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::TransferToServerTransformMetric() {
  static metrics::Metric* metric = new metrics::Metric(
      "TransferToServerTransformTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::TransferFromServerMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("TransferFromServerTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::CompileMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("CompileTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::ExecuteMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ExecuteTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::ExecuteReplicatedMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ExecuteReplicatedTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::ExecuteParallelMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ExecuteParallelTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::ExecuteChainedMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ExecuteChainedTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::DeconstructTupleMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("DeconstructTupleTime", metrics::MetricFnTime);
  return metric;
}

metrics::Counter* ComputationClient::CreateAsyncDataHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter =
      new metrics::Counter("CreateAsyncDataHandles");
  return counter;
}

metrics::Counter* ComputationClient::CreateDataHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter = new metrics::Counter("CreateDataHandles");
  return counter;
}

metrics::Counter* ComputationClient::ReleaseDataHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter = new metrics::Counter("ReleaseDataHandles");
  return counter;
}

metrics::Counter* ComputationClient::DestroyDataHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter = new metrics::Counter("DestroyDataHandles");
  return counter;
}

metrics::Metric* ComputationClient::ReleaseDataHandlesTimeMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ReleaseDataHandlesTime", metrics::MetricFnTime);
  return metric;
}

metrics::Counter* ComputationClient::CreateCompileHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter =
      new metrics::Counter("CreateCompileHandles");
  return counter;
}

metrics::Counter* ComputationClient::ReleaseCompileHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter =
      new metrics::Counter("ReleaseCompileHandles");
  return counter;
}

metrics::Counter* ComputationClient::DestroyCompileHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter =
      new metrics::Counter("DestroyCompileHandles");
  return counter;
}

metrics::Metric* ComputationClient::ReleaseCompileHandlesTimeMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ReleaseCompileHandlesTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::InboundDataMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("InboundData", metrics::MetricFnBytes);
  return metric;
}

metrics::Metric* ComputationClient::OutboundDataMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("OutboundData", metrics::MetricFnBytes);
  return metric;
}

}  // namespace xla
