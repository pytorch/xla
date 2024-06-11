#include "torch_xla/csrc/runtime/computation_client.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "tsl/platform/stacktrace_handler.h"
#include "xla/status_macros.h"

namespace torch_xla {
namespace runtime {

std::shared_ptr<ComputationClient::Computation> ComputationClient::Compile(
    xla::XlaComputation computation, std::string compilation_device,
    std::vector<std::string> devices, const xla::Shape* output_shape) {
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

metrics::Metric* ComputationClient::TransferToDeviceMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("TransferToDeviceTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::TransferToDeviceTransformMetric() {
  static metrics::Metric* metric = new metrics::Metric(
      "TransferToDeviceTransformTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::TransferFromDeviceMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("TransferFromDeviceTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::CompileMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("CompileTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::EagerCompileMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("EagerOpCompileTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::ExecuteMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ExecuteTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::EagerExecuteMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("EagerOpExecuteTime", metrics::MetricFnTime);
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

metrics::Counter* ComputationClient::StableHloCompileCounter() {
  static metrics::Counter* counter = new metrics::Counter("StableHloCompile");
  return counter;
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

}  // namespace runtime
}  // namespace torch_xla
