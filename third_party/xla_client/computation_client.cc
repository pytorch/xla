#include "tensorflow/compiler/xla/xla_client/computation_client.h"

#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"

namespace xla {
namespace {

string GetTpuClusterConfigPath() {
  string home_folder = sys_util::GetEnvString("HOME", ".");
  return absl::StrCat(home_folder, "/", ".pytorch_tpu.conf");
}

bool HasXrtConfigFile(string* config_path) {
  *config_path = GetTpuClusterConfigPath();
  if (access(config_path->c_str(), F_OK) != -1) {
    // If we have a TPU cluster config file, we are in Cloud TPU world, so steer
    // towards config file based XRT client.
    return true;
  }
  config_path->clear();
  return false;
}

XrtComputationClient::Worker ParseWorker(const string& worker) {
  std::vector<string> parts = absl::StrSplit(worker, ':');
  XLA_CHECK(parts.size() == 1 || parts.size() == 2) << worker;
  return parts.size() == 1
             ? XrtComputationClient::Worker(parts[0], 0)
             : XrtComputationClient::Worker(parts[0], std::stoi(parts[1]));
}

void AddXrtHostDevices(const string& worker_name, int task_no,
                       const string& server,
                       std::map<string, int>* device_ordinals,
                       XrtComputationClient::Options* options) {
  struct Devices {
    const char* name;
    int count;
  } const devices[] = {
      {"TPU", sys_util::GetEnvInt("TPU_NUM_DEVICES", 8)},
      {"CPU", sys_util::GetEnvInt("CPU_NUM_DEVICES", 1)},
  };
  string host_port = server.compare(0, 7, "grpc://") == 0
                         ? server
                         : absl::StrCat("grpc://", server);
  options->workers_map.emplace(
      XrtComputationClient::Worker(worker_name, task_no), host_port);
  for (auto& device : devices) {
    int& device_ordinal = (*device_ordinals)[device.name];
    for (int j = 0; j < device.count; ++j, ++device_ordinal) {
      string device_name = absl::StrCat(device.name, ":", device_ordinal);
      string xrt_device_name =
          absl::StrCat("/job:", worker_name, "/replica:0/task:", task_no,
                       "/device:", device_name);
      options->device_map.emplace(device_name, xrt_device_name);
    }
  }
}

StatusOr<bool> ParseEnvBasedTpuClusterConfig(
    XrtComputationClient::Options* options) {
  string tpu_config = sys_util::GetEnvString("XRT_TPU_CONFIG", "");
  if (tpu_config.empty()) {
    return false;
  }
  std::map<string, int> device_ordinals;
  std::vector<string> spec_parts = absl::StrSplit(tpu_config, '|');
  TF_RET_CHECK(!spec_parts.empty()) << tpu_config;
  for (const auto& spec : spec_parts) {
    std::vector<string> host_parts = absl::StrSplit(spec, ';');
    TF_RET_CHECK(host_parts.size() == 3) << spec;
    AddXrtHostDevices(host_parts[0], std::stoi(host_parts[1]), host_parts[2],
                      &device_ordinals, options);
  }
  options->default_device = "TPU:0";
  return true;
}

Status ParseTpuClusterConfig(const string& xrt_config_path,
                             XrtComputationClient::Options* options) {
  std::map<string, int> device_ordinals;
  std::ifstream config_file(xrt_config_path);
  string line;
  while (std::getline(config_file, line)) {
    if (line.compare(0, 7, "worker:") == 0) {
      std::vector<string> parts =
          absl::StrSplit(line.substr(7), ' ', absl::SkipWhitespace());
      TF_RET_CHECK(parts.size() >= 2) << line;
      const string& worker_name = parts[0];
      for (std::size_t i = 1; i < parts.size(); ++i) {
        AddXrtHostDevices(worker_name, i - 1, parts[i], &device_ordinals,
                          options);
      }
    }
  }
  options->default_device = "TPU:0";
  return Status::OK();
}

}  // namespace

StatusOr<std::unique_ptr<ComputationClient>> ComputationClient::Create() {
  XrtComputationClient::Options options;
  string xrt_config_path;
  if (HasXrtConfigFile(&xrt_config_path)) {
    TF_LOG(INFO) << "Loading XRT configuration from " << xrt_config_path;
    TF_RETURN_IF_ERROR(ParseTpuClusterConfig(xrt_config_path, &options));
  } else {
    TF_ASSIGN_OR_RETURN(bool configured,
                        ParseEnvBasedTpuClusterConfig(&options));
    if (!configured) {
      string device_spec = sys_util::GetEnvString(
          "XRT_DEVICE_MAP",
          "TPU:0;/job:tpu_worker/replica:0/task:0/device:TPU:0");
      for (const auto& device_target : absl::StrSplit(device_spec, '|')) {
        std::vector<string> parts = absl::StrSplit(device_target, ';');
        TF_RET_CHECK(parts.size() == 2) << device_target;
        if (options.default_device.empty()) {
          options.default_device = parts[0];
        }
        options.device_map.emplace(parts[0], parts[1]);
      }
      string workers_spec = sys_util::GetEnvString(
          "XRT_WORKERS", "tpu_worker:0;grpc://localhost:51000");
      for (const auto& name_target : absl::StrSplit(workers_spec, '|')) {
        std::vector<string> parts = absl::StrSplit(name_target, ';');
        TF_RET_CHECK(parts.size() == 2);
        options.workers_map.emplace(ParseWorker(parts[0]), parts[1]);
      }
    }
  }
  return std::unique_ptr<ComputationClient>(new XrtComputationClient(options));
}

std::shared_ptr<ComputationClient::Computation> ComputationClient::Compile(
    XlaComputation computation, std::vector<string> devices,
    const Shape* output_shape) {
  std::vector<CompileInstance> instances;
  instances.emplace_back(std::move(computation), std::move(devices),
                         output_shape);
  std::vector<std::shared_ptr<Computation>> results =
      Compile(std::move(instances));
  return std::move(results[0]);
}

int64 ComputationClient::GetDeviceOrdinal(const string& device) {
  auto pos = device.rfind(':');
  CHECK_NE(pos, string::npos) << device;
  return std::stoi(device.substr(pos + 1));
}

metrics::Metric* ComputationClient::TransferToServerMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("TransferToServerTime", metrics::MetricFnTime);
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

metrics::Metric* ComputationClient::DeconstructTupleMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("DeconstructTupleTime", metrics::MetricFnTime);
  return metric;
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
