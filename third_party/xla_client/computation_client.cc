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
#include "xla/status_macros.h"
#include "tsl/platform/net.h"
#include "tsl/platform/stacktrace_handler.h"
#include "tsl/util/device_name_utils.h"
#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/env_vars.h"
#include "third_party/xla_client/mesh_service.h"
#include "third_party/xla_client/pjrt_computation_client.h"
#include "third_party/xla_client/sys_util.h"

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
    tsl::testing::InstallStacktraceHandler();
  }
  auto client = ComputationClient::Create();
  return client.release();
}

std::string MakeGrpcEndPoint(const std::string& server) {
  return server.compare(0, 7, "grpc://") == 0 ? server
                                              : absl::StrCat("grpc://", server);
}

std::string BuildTaskDeviceKey(int task_no, const std::string& kind) {
  return absl::StrCat(task_no, ":", kind);
}
}  // namespace

std::unique_ptr<ComputationClient> ComputationClient::Create() {
  std::unique_ptr<ComputationClient> client;

  if (sys_util::GetEnvString(env::kEnvPjRtDevice, "") != "") {
    client = std::unique_ptr<ComputationClient>(new PjRtComputationClient());
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
