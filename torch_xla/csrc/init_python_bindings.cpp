#include <Python.h>
#include <c10/core/Device.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>

#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/types/variant.h"
#include "pybind11/attr.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl_bind.h"
#include "tensorflow/compiler/xla/pjrt/distributed/distributed.h"
#include "tensorflow/compiler/xla/python/profiler/internal/traceme_wrapper.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/python/profiler/internal/profiler_pywrap_impl.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"
#include "third_party/xla_client/computation_client.h"
#include "third_party/xla_client/mesh_service.h"
#include "third_party/xla_client/metrics.h"
#include "third_party/xla_client/metrics_analysis.h"
#include "third_party/xla_client/metrics_reader.h"
#include "third_party/xla_client/multi_wait.h"
#include "third_party/xla_client/profiler.h"
#include "third_party/xla_client/record_reader.h"
#include "third_party/xla_client/sys_util.h"
#include "third_party/xla_client/thread_pool.h"
#include "third_party/xla_client/util.h"
#include "third_party/xla_client/xla_util.h"
#include "torch_xla/csrc/XLANativeFunctions.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/computation.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/ir_util.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/tensor_impl.h"
#include "torch_xla/csrc/tensor_methods.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/version.h"
#include "torch_xla/csrc/xla_backend_impl.h"
#include "torch_xla/csrc/xla_graph_executor.h"
#include "torch_xla/csrc/xla_op_builder.h"
#include "torch_xla/csrc/xla_sharding_util.h"

namespace torch_xla {
namespace {

static int64_t seed_info_id = -127389;

struct NoGilSection {
  NoGilSection() : state(PyEval_SaveThread()) {}
  ~NoGilSection() { PyEval_RestoreThread(state); }
  PyThreadState* state = nullptr;
};

c10::optional<torch::lazy::BackendDevice> GetOptionalDevice(
    const std::string& device_str) {
  if (device_str.empty()) {
    return c10::nullopt;
  }
  return bridge::AtenDeviceToXlaDevice(c10::Device(device_str));
}

torch::lazy::BackendDevice GetDeviceOrCurrent(const std::string& device_str) {
  if (device_str.empty()) {
    return GetCurrentDevice();
  }
  return bridge::AtenDeviceToXlaDevice(c10::Device(device_str));
}

void PrepareToExit() {
  xla::ComputationClient* client = xla::ComputationClient::GetIfInitialized();
  if (client != nullptr) {
    XLAGraphExecutor::Get()->WaitDeviceOps({});
    client->PrepareToExit();
  }
}

std::string GetTensorsDump(
    const std::vector<at::Tensor>& tensors,
    const std::function<
        std::string(absl::Span<const torch::lazy::Node* const>)>& coverter) {
  std::vector<const torch::lazy::Node*> nodes;
  std::vector<torch::lazy::Value> values;
  for (auto& tensor : tensors) {
    XLATensorPtr xtensor = bridge::GetXlaTensor(tensor);
    values.push_back(xtensor->GetIrValue());
    nodes.push_back(values.back().node.get());
  }
  return coverter(nodes);
}

std::string SetCurrentThreadDevice(const std::string& device_str) {
  c10::Device prev_device = bridge::SetCurrentDevice(c10::Device(device_str));
  std::stringstream ss;
  ss << prev_device;
  return ss.str();
}

std::string GetCurrentThreadDevice() {
  return bridge::GetCurrentAtenDevice().str();
}

std::vector<std::string> GetXlaDevices(
    const std::vector<std::string>& devices) {
  std::vector<std::string> xla_devices;
  xla_devices.reserve(devices.size());
  for (auto& device_str : devices) {
    torch::lazy::BackendDevice device =
        bridge::AtenDeviceToXlaDevice(c10::Device(device_str));
    xla_devices.emplace_back(device.toString());
  }
  return xla_devices;
}

std::vector<XLATensorPtr> GetXlaTensors(const std::vector<at::Tensor>& tensors,
                                        bool want_all) {
  std::vector<XLATensorPtr> xtensors;
  xtensors.reserve(tensors.size());
  if (want_all) {
    for (auto& tensor : tensors) {
      xtensors.push_back(bridge::GetXlaTensor(tensor));
    }
  } else {
    for (auto& tensor : tensors) {
      auto xtensor = bridge::TryGetXlaTensor(tensor);
      if (xtensor) {
        xtensors.push_back(xtensor);
      }
    }
  }
  return xtensors;
}

AllReduceType GetReduceType(const std::string& reduce_type) {
  if (reduce_type == "sum") {
    return AllReduceType::kSum;
  } else if (reduce_type == "mul") {
    return AllReduceType::kMul;
  } else if (reduce_type == "and") {
    return AllReduceType::kAnd;
  } else if (reduce_type == "or") {
    return AllReduceType::kOr;
  } else if (reduce_type == "min") {
    return AllReduceType::kMin;
  } else if (reduce_type == "max") {
    return AllReduceType::kMax;
  }
  XLA_ERROR() << "Unknown AllReduce type: " << reduce_type;
}

std::vector<std::vector<int64_t>> CreateReduceGroups(const py::list& groups) {
  std::vector<std::vector<int64_t>> replica_groups;
  for (auto& group : groups) {
    replica_groups.emplace_back();
    for (auto& replica_id : group.cast<py::list>()) {
      replica_groups.back().push_back(replica_id.cast<int64_t>());
    }
  }
  return replica_groups;
}

std::vector<std::pair<int64_t, int64_t>> CreateSourceTargetPairs(
    const py::list& pairs) {
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs;
  for (auto& pair : pairs) {
    const auto& pylist_pair = pair.cast<py::list>();
    XLA_CHECK_EQ(len(pylist_pair), 2);
    source_target_pairs.push_back(
        {pylist_pair[0].cast<int64_t>(), pylist_pair[1].cast<int64_t>()});
  }
  return source_target_pairs;
}

std::shared_ptr<torch::lazy::Value> AllReduceInPlace(
    const std::string& reduce_type, const std::vector<at::Tensor>& tensors,
    const std::shared_ptr<torch::lazy::Value>& token, double scale,
    const std::vector<std::vector<int64_t>>& replica_groups, bool pin_layout) {
  std::vector<XLATensorPtr> xtensors =
      GetXlaTensors(tensors, /*want_all=*/true);
  return std::make_shared<torch::lazy::Value>(
      tensor_methods::all_reduce(&xtensors, *token, GetReduceType(reduce_type),
                                 scale, replica_groups, pin_layout));
}

std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>> AllReduce(
    const std::string& reduce_type, const at::Tensor& input,
    const std::shared_ptr<torch::lazy::Value>& token, double scale,
    const std::vector<std::vector<int64_t>>& replica_groups, bool pin_layout) {
  XLATensorPtr result;
  torch::lazy::Value new_token;
  std::tie(result, new_token) = tensor_methods::all_reduce(
      bridge::GetXlaTensor(input), *token, GetReduceType(reduce_type), scale,
      replica_groups, pin_layout);
  return std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>>(
      bridge::AtenFromXlaTensor(std::move(result)),
      std::make_shared<torch::lazy::Value>(new_token));
}

std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>> ReduceScatter(
    const std::string& reduce_type, const at::Tensor& input,
    const std::shared_ptr<torch::lazy::Value>& token, double scale,
    int64_t scatter_dim, int64_t shard_count,
    const std::vector<std::vector<int64_t>>& replica_groups, bool pin_layout) {
  XLATensorPtr result;
  torch::lazy::Value new_token;
  std::tie(result, new_token) = tensor_methods::reduce_scatter(
      bridge::GetXlaTensor(input), *token, GetReduceType(reduce_type), scale,
      scatter_dim, shard_count, replica_groups, pin_layout);
  return std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>>(
      bridge::AtenFromXlaTensor(std::move(result)),
      std::make_shared<torch::lazy::Value>(new_token));
}

std::shared_ptr<torch::lazy::Value> ReduceScatterOut(
    const std::string& reduce_type, at::Tensor& output, const at::Tensor& input,
    const std::shared_ptr<torch::lazy::Value>& token, double scale,
    int64_t scatter_dim, int64_t shard_count,
    const std::vector<std::vector<int64_t>>& replica_groups, bool pin_layout) {
  XLATensorPtr out = bridge::GetXlaTensor(output);
  torch::lazy::Value new_token;
  new_token = tensor_methods::reduce_scatter_out(
      out, bridge::GetXlaTensor(input), *token, GetReduceType(reduce_type),
      scale, scatter_dim, shard_count, replica_groups, pin_layout);
  return std::make_shared<torch::lazy::Value>(new_token);
}

std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>> AllGather(
    const at::Tensor& input, const std::shared_ptr<torch::lazy::Value>& token,
    int64_t dim, int64_t shard_count,
    const std::vector<std::vector<int64_t>>& replica_groups, bool pin_layout) {
  XLATensorPtr result;
  torch::lazy::Value new_token;
  std::tie(result, new_token) =
      tensor_methods::all_gather(bridge::GetXlaTensor(input), *token, dim,
                                 shard_count, replica_groups, pin_layout);
  return {bridge::AtenFromXlaTensor(std::move(result)),
          std::make_shared<torch::lazy::Value>(new_token)};
}

std::shared_ptr<torch::lazy::Value> AllGatherOut(
    at::Tensor& output, const at::Tensor& input,
    const std::shared_ptr<torch::lazy::Value>& token, int64_t dim,
    int64_t shard_count,
    const std::vector<std::vector<int64_t>>& replica_groups, bool pin_layout) {
  XLATensorPtr out = bridge::GetXlaTensor(output);
  torch::lazy::Value new_token;
  new_token = tensor_methods::all_gather_out(out, bridge::GetXlaTensor(input),
                                             *token, dim, shard_count,
                                             replica_groups, pin_layout);
  return std::make_shared<torch::lazy::Value>(new_token);
}

std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>> AllToAll(
    const at::Tensor& input, const std::shared_ptr<torch::lazy::Value>& token,
    int64_t split_dimension, int64_t concat_dimension, int64_t split_count,
    const std::vector<std::vector<int64_t>>& replica_groups, bool pin_layout) {
  XLATensorPtr result;
  torch::lazy::Value new_token;
  std::tie(result, new_token) = tensor_methods::all_to_all(
      bridge::GetXlaTensor(input), *token, split_dimension, concat_dimension,
      split_count, replica_groups, pin_layout);
  return std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>>(
      bridge::AtenFromXlaTensor(std::move(result)),
      std::make_shared<torch::lazy::Value>(new_token));
}

std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>> CollectivePermute(
    const at::Tensor& input, const std::shared_ptr<torch::lazy::Value>& token,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs) {
  XLATensorPtr result;
  torch::lazy::Value new_token;
  std::tie(result, new_token) = tensor_methods::collective_permute(
      bridge::GetXlaTensor(input), *token, source_target_pairs);
  return std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>>(
      bridge::AtenFromXlaTensor(std::move(result)),
      std::make_shared<torch::lazy::Value>(new_token));
}

void OptimizationBarrier_(std::vector<at::Tensor>& tensors) {
  std::vector<XLATensorPtr> xtensors =
      GetXlaTensors(tensors, /*want_all=*/false);
  tensor_methods::optimization_barrier_(xtensors);
}

std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>> Send(
    const at::Tensor& input, const std::shared_ptr<torch::lazy::Value>& token,
    int64_t channel_id) {
  XLATensorPtr result;
  torch::lazy::Value new_token;
  std::tie(result, new_token) =
      tensor_methods::send(bridge::GetXlaTensor(input), *token, channel_id);
  return {bridge::AtenFromXlaTensor(std::move(result)),
          std::make_shared<torch::lazy::Value>(new_token)};
}

std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>> Recv(
    at::Tensor& output, const std::shared_ptr<torch::lazy::Value>& token,
    int64_t channel_id) {
  XLATensorPtr out = bridge::GetXlaTensor(output);
  XLATensorPtr result;
  torch::lazy::Value new_token;
  std::tie(result, new_token) = tensor_methods::recv(out, *token, channel_id);
  return {bridge::AtenFromXlaTensor(std::move(result)),
          std::make_shared<torch::lazy::Value>(new_token)};
}

void SyncTensors(const std::vector<at::Tensor>& tensors,
                 const std::vector<std::string>& devices, bool wait,
                 bool sync_xla_data, bool warm_up_cache_only = false) {
  std::vector<XLATensorPtr> xtensors =
      GetXlaTensors(tensors, /*want_all=*/false);
  XLAGraphExecutor::Get()->SyncTensorsGraph(&xtensors, devices, wait,
                                            sync_xla_data, warm_up_cache_only);
}

void SyncLiveTensors(const std::string& device_str,
                     const std::vector<std::string>& devices, bool wait) {
  auto opt_device = GetOptionalDevice(device_str);
  XLAGraphExecutor::Get()->SyncLiveTensorsGraph(
      opt_device ? &opt_device.value() : nullptr, devices, wait);
}

void StepMarker(const std::string& device_str,
                const std::vector<std::string>& devices, bool wait) {
  tsl::profiler::TraceMe activity("StepMarker",
                                  tsl::profiler::TraceMeLevel::kInfo);
  torch::lazy::BackendDevice device = GetDeviceOrCurrent(device_str);
  XLAGraphExecutor::Get()->SyncLiveTensorsGraph(&device, devices, wait);
  XLAGraphExecutor::Get()->MarkStep(device);
  bool debug_mode = xla::sys_util::GetEnvBool("PT_XLA_DEBUG", false);
  if (TF_PREDICT_FALSE(debug_mode)) {
    std::string report = xla::metrics::CreatePerformanceReport(
        xla::ComputationClient::Get()->GetMetrics());
    if (!report.empty()) {
      std::string fout = xla::sys_util::GetEnvString("PT_XLA_DEBUG_FILE", "");
      if (TF_PREDICT_FALSE(!fout.empty())) {
        std::ofstream out_file(fout, std::ios_base::app);
        out_file << report;
      } else {
        std::cout << report;
      }
    }
  }
}

void SetRngSeed(uint64_t seed, const std::string& device_str) {
  torch::lazy::BackendDevice device = GetDeviceOrCurrent(device_str);
  XLAGraphExecutor::Get()->SetRngSeed(device, seed);
}

uint64_t GetRngSeed(const std::string& device_str) {
  return XLAGraphExecutor::Get()->GetRunningSeed(
      GetDeviceOrCurrent(device_str));
}

std::string GetTensorsHloGraph(const std::vector<at::Tensor>& tensors) {
  std::vector<XLATensorPtr> xtensors =
      GetXlaTensors(tensors, /*want_all=*/false);
  return XLAGraphExecutor::Get()->DumpHloComputation(xtensors);
}

std::string GetLiveTensorsReport(size_t nodes_threshold,
                                 const std::string& device_str) {
  auto opt_device = GetOptionalDevice(device_str);
  auto tensors = XLAGraphExecutor::Get()->GetLiveTensors(
      opt_device ? &opt_device.value() : nullptr);
  std::stringstream ss;
  for (auto& tensor : tensors) {
    torch::lazy::Value ir_value = tensor->CurrentIrValue();
    if (ir_value) {
      std::vector<const torch::lazy::Node*> roots({ir_value.node.get()});
      auto post_order = torch::lazy::Util::ComputePostOrder(roots);
      if (post_order.size() > nodes_threshold) {
        ss << "Tensor: id=" << tensor->GetUniqueId()
           << ", shape=" << tensor->shape().get()
           << ", device=" << tensor->GetDevice()
           << ", ir_nodes=" << post_order.size() << "\n";
        for (size_t i = post_order.size(); i > 0; --i) {
          if (!post_order[i - 1]->metadata().frame_info.empty()) {
            ss << post_order[i - 1]->metadata().frame_info;
            break;
          }
        }
        ss << DumpUtil::PostOrderToText(post_order, roots);
        ss << "\n\n";
      }
    }
  }
  return ss.str();
}

void ClearPendingIrs(const std::string& device_str) {
  auto opt_device = GetOptionalDevice(device_str);
  XLA_CHECK(opt_device);
  auto tensors = XLAGraphExecutor::Get()->GetLiveTensors(&opt_device.value());
  XLAGraphExecutor::Get()->ClearPendingIrs(tensors, opt_device.value());
}

std::ptrdiff_t GetTensorViewAliasId(const at::Tensor& tensor) {
  XLATensorPtr xtensor = bridge::GetXlaTensor(tensor);
  return xtensor->GetViewAliasId();
}

std::ptrdiff_t GetTensorId(const at::Tensor& tensor) {
  XLATensorPtr xtensor = bridge::GetXlaTensor(tensor);
  return xtensor->GetUniqueId();
}

std::vector<at::Tensor> GetXlaTensorsFromAten(
    const std::vector<at::Tensor>& aten_tensors,
    const std::vector<std::string>& devices) {
  auto data_handles = CreateTensorsData(aten_tensors, GetXlaDevices(devices));

  std::vector<at::Tensor> xla_tensors;
  xla_tensors.reserve(data_handles.size());
  for (auto& data_handle : data_handles) {
    XLATensorPtr xla_tensor = XLATensor::Create(std::move(data_handle));
    xla_tensors.push_back(bridge::AtenFromXlaTensor(std::move(xla_tensor)));
  }
  return xla_tensors;
}

std::shared_ptr<torch::lazy::Value> CreateToken(const std::string& device_str) {
  // This should be using xla::CreateToken() once we have added Token support to
  // XLA AllReduce(). Meanwhile we use a constant as token, and we handle it
  // accordingly in cross_replica_reduces.cpp.
  // This needs to be device data (hence coming in as XLA computation parameter)
  // as otherwise the XLA compiler passes will remove it, vanishing its
  // sequencing effects.
  torch::lazy::BackendDevice device = GetDeviceOrCurrent(device_str);
  torch::lazy::Value ir_value = XLAGraphExecutor::Get()->GetDeviceDataIrValue(
      0.0, xla::PrimitiveType::F32, device);
  return std::make_shared<torch::lazy::Value>(std::move(ir_value));
}

at::Tensor GetXlaTensorDimensionSize(const at::Tensor& tensor, int64_t dim) {
  XLATensorPtr xtensor = bridge::GetXlaTensor(tensor);
  return bridge::AtenFromXlaTensor(
      tensor_methods::get_dimensions_size(xtensor, {dim}));
}

template <class T>
py::object GetMetricData(const T* data) {
  double accumulator = 0.0;
  size_t total_samples = 0;
  auto samples = data->Samples(&accumulator, &total_samples);
  auto py_samples = py::tuple(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    auto sample = py::tuple(2);
    sample[0] = 1.0e-9 * samples[i].timestamp_ns;
    sample[1] = samples[i].value;

    py_samples[i] = sample;
  }
  auto result = py::tuple(3);
  result[0] = total_samples;
  result[1] = accumulator;
  result[2] = py_samples;
  return result;
}

py::object GetMetricData(const std::string& name) {
  if (auto* data = torch::lazy::GetMetric(name)) {
    return GetMetricData<torch::lazy::MetricData>(data);
  }
  if (auto* data = xla::metrics::GetMetric(name)) {
    return GetMetricData<xla::metrics::MetricData>(data);
  }
  return py::none();
}

py::object GetRevisions() {
  auto py_dict = py::dict();
  py_dict["xla"] = std::string(XLA_GITREV);
  py_dict["torch"] = std::string(TORCH_GITREV);
  return py_dict;
}

std::vector<py::bytes> Rendezvous(int ordinal, const std::string& tag,
                                  const std::string& payload,
                                  const std::vector<int64_t>& replicas) {
  xla::service::MeshClient* mesh_client = xla::service::MeshClient::Get();
  std::vector<py::bytes> payloads;
  if (mesh_client != nullptr) {
    auto rendezvous_payloads =
        mesh_client->Rendezvous(ordinal, tag, payload, replicas);
    for (auto& rendezvous_payload : rendezvous_payloads) {
      payloads.push_back(rendezvous_payload);
    }
  } else {
    XLA_CHECK(replicas.empty() || (replicas.size() == 1 && replicas[0] == 0));
  }
  return payloads;
}

std::shared_ptr<xla::util::RecordReader> CreateRecordReader(
    std::string path, const std::string& compression, int64_t buffer_size) {
  return std::make_shared<xla::util::RecordReader>(std::move(path), compression,
                                                   buffer_size);
}

bool RecordRead(const std::shared_ptr<xla::util::RecordReader>& reader,
                xla::util::RecordReader::Data* value) {
  NoGilSection nogil;
  return reader->Read(value);
}

py::object RecordReadExample(
    const std::shared_ptr<xla::util::RecordReader>& reader) {
  auto make_r1_size = [](int64_t size) -> std::vector<int64_t> {
    return std::vector<int64_t>({size});
  };

  xla::util::RecordReader::Data value;
  if (!RecordRead(reader, &value)) {
    return py::none();
  }
  tensorflow::Example exmsg;
  if (!exmsg.ParseFromArray(value.data(), value.size())) {
    XLA_ERROR() << "Unable to parse TF example from " << reader->path();
  }
  auto example = py::dict();
  for (auto& name_feat : exmsg.features().feature()) {
    switch (name_feat.second.kind_case()) {
      case tensorflow::Feature::kBytesList: {
        const tensorflow::BytesList& bvalue = name_feat.second.bytes_list();
        if (bvalue.value_size() == 1) {
          const std::string& svalue = bvalue.value(0);
          at::Tensor data = at::empty(make_r1_size(svalue.size()),
                                      at::TensorOptions(at::kChar));
          std::memcpy(data.data_ptr<int8_t>(), svalue.data(), svalue.size());
          example[py::str(name_feat.first)] =
              torch::autograd::make_variable(data);
        } else {
          auto tlist = py::list(bvalue.value_size());
          for (int i = 0; i < bvalue.value_size(); ++i) {
            const std::string& svalue = bvalue.value(i);
            at::Tensor data = at::empty(make_r1_size(svalue.size()),
                                        at::TensorOptions(at::kChar));
            std::memcpy(data.data_ptr<int8_t>(), svalue.data(), svalue.size());
            tlist[i] = torch::autograd::make_variable(data);
          }
          example[py::str(name_feat.first)] = tlist;
        }
      } break;
      case tensorflow::Feature::kFloatList: {
        const tensorflow::FloatList& fvalue = name_feat.second.float_list();
        at::Tensor data = at::empty(make_r1_size(fvalue.value_size()),
                                    at::TensorOptions(at::kFloat));
        std::memcpy(data.data_ptr<float>(), fvalue.value().data(),
                    fvalue.value_size() * sizeof(float));
        example[py::str(name_feat.first)] =
            torch::autograd::make_variable(data);
      } break;
      case tensorflow::Feature::kInt64List: {
        const tensorflow::Int64List& ivalue = name_feat.second.int64_list();
        at::Tensor data = at::empty(make_r1_size(ivalue.value_size()),
                                    at::TensorOptions(at::kLong));
        std::memcpy(data.data_ptr<int64_t>(), ivalue.value().data(),
                    ivalue.value_size() * sizeof(int64_t));
        example[py::str(name_feat.first)] =
            torch::autograd::make_variable(data);
      } break;
      default:
        XLA_ERROR() << "Unknown data type from " << reader->path();
    }
  }
  return example;
}

std::unique_ptr<tsl::RandomAccessFile> OpenTfFile(const std::string& path) {
  tsl::Env* env = tsl::Env::Default();
  std::unique_ptr<tsl::RandomAccessFile> file;
  XLA_CHECK_OK(env->NewRandomAccessFile(path, &file));
  return file;
}

py::object StatTfFile(const std::string& path) {
  tsl::Env* env = tsl::Env::Default();
  tsl::FileStatistics stat;
  {
    NoGilSection nogil;
    XLA_CHECK_OK(env->Stat(path, &stat));
  }
  auto py_stat = py::dict();
  py_stat["length"] = py::cast(stat.length);
  py_stat["mtime_nsec"] = py::cast(stat.mtime_nsec);
  py_stat["is_directory"] = py::cast(stat.is_directory);
  return py_stat;
}

py::bytes ReadTfFile(tsl::RandomAccessFile* file, uint64_t offset,
                     size_t size) {
  static const size_t kMinReadSize = 1024 * 1024;
  std::unique_ptr<char[]> buffer;
  {
    NoGilSection nogil;
    buffer.reset(new char[size]);

    size_t num_threads = std::max<size_t>(size / kMinReadSize, 1);
    num_threads =
        std::min<size_t>(num_threads, std::thread::hardware_concurrency());
    size_t block_size = size / num_threads;

    auto mwait = std::make_shared<xla::util::MultiWait>(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      auto reader = [&, i]() {
        uint64_t base = static_cast<uint64_t>(i) * block_size;
        size_t tsize =
            (i + 1 < num_threads) ? block_size : (size - i * block_size);

        tsl::StringPiece result;
        XLA_CHECK_OK(
            file->Read(offset + base, tsize, &result, buffer.get() + base));
      };
      xla::env::ScheduleIoClosure(
          xla::util::MultiWait::Completer(mwait, std::move(reader)));
    }
    mwait->Wait();
  }
  return py::bytes(buffer.get(), size);
}

std::unique_ptr<tsl::WritableFile> CreateTfFile(const std::string& path) {
  tsl::Env* env = tsl::Env::Default();
  std::unique_ptr<tsl::WritableFile> file;
  XLA_CHECK_OK(env->NewWritableFile(path, &file));
  return file;
}

void WriteTfFile(tsl::WritableFile* file, const std::string& data) {
  XLA_CHECK_OK(file->Append(tsl::StringPiece(data.data(), data.size())));
}

void FlushTfFile(tsl::WritableFile* file) {
  XLA_CHECK_OK(file->Flush());
  XLA_CHECK_OK(file->Sync());
}

py::object ListTfFs(const std::string& pattern) {
  std::vector<std::string> files;
  {
    NoGilSection nogil;
    tsl::Env* env = tsl::Env::Default();
    XLA_CHECK_OK(env->GetMatchingPaths(pattern, &files));
  }

  auto py_files = py::tuple(files.size());
  for (size_t i = 0; i < files.size(); ++i) {
    py_files[i] = files[i];
  }
  return py_files;
}

void RemoveTfFile(const std::string& path) {
  tsl::Env* env = tsl::Env::Default();
  XLA_CHECK_OK(env->DeleteFile(path));
}

py::object XlaNms(const at::Tensor& boxes, const at::Tensor& scores,
                  const at::Tensor& score_threshold,
                  const at::Tensor& iou_threshold, int64_t output_size) {
  at::Tensor selected_indices;
  at::Tensor num_valid;
  {
    NoGilSection nogil;
    auto nms_result = tensor_methods::nms(
        bridge::GetXlaTensor(boxes), bridge::GetXlaTensor(scores),
        bridge::GetXlaTensor(score_threshold),
        bridge::GetXlaTensor(iou_threshold), output_size);
    selected_indices = bridge::AtenFromXlaTensor(std::move(nms_result.first));
    num_valid = bridge::AtenFromXlaTensor(std::move(nms_result.second));
  }
  auto result_tuple = py::tuple(2);
  result_tuple[0] =
      torch::autograd::make_variable(selected_indices, /*requires_grad=*/false);
  result_tuple[1] =
      torch::autograd::make_variable(num_valid, /*requires_grad=*/false);
  return result_tuple;
}

std::vector<at::Tensor> XlaUserComputation(
    const std::string& opname, const std::vector<at::Tensor>& inputs,
    ComputationPtr computation) {
  std::vector<XLATensorPtr> xinputs = GetXlaTensors(inputs, /*want_all=*/true);
  std::vector<XLATensorPtr> xresults =
      tensor_methods::user_computation(opname, xinputs, std::move(computation));
  std::vector<at::Tensor> results;
  for (auto& xresult : xresults) {
    at::Tensor tensor = bridge::AtenFromXlaTensor(std::move(xresult));
    results.push_back(
        torch::autograd::make_variable(tensor, /*requires_grad=*/false));
  }
  return results;
}

ComputationPtr CreateComputation(const std::string& name, xla::XlaOp root) {
  xla::XlaComputation computation = ConsumeValue(root.builder()->Build(root));
  return std::make_shared<Computation>(name, std::move(computation));
}

ComputationPtr CreateComputationFromProto(const std::string& name,
                                          const std::string& module_proto) {
  xla::HloModuleProto proto;
  proto.ParseFromString(module_proto);
  xla::XlaComputation computation(std::move(proto));
  return std::make_shared<Computation>(name, std::move(computation));
}

xla::Shape GetTensorShape(const at::Tensor& tensor,
                          const std::string& device_str) {
  auto xtensor = bridge::TryGetXlaTensor(tensor);
  if (xtensor) {
    return xtensor->shape();
  }
  torch::lazy::BackendDevice device = GetDeviceOrCurrent(device_str);
  return CreateComputationShapeFromTensor(tensor, &device);
}

py::dict GetMemoryInfo(const std::string& device_str) {
  xla::ComputationClient::MemoryInfo mem_info;
  {
    NoGilSection nogil;
    torch::lazy::BackendDevice device = GetDeviceOrCurrent(device_str);
    mem_info = xla::ComputationClient::Get()->GetMemoryInfo(device.toString());
  }
  auto py_dict = py::dict();
  py_dict["kb_free"] = mem_info.kb_free;
  py_dict["kb_total"] = mem_info.kb_total;
  return py_dict;
}

// Must be called holding GIL as it reads Python objects. Also, Python objects
// are reference counted; reading py::dict will increase its reference count.
absl::flat_hash_map<std::string, std::variant<int, std::string>>
ConvertDictToMap(const py::dict& dictionary) {
  absl::flat_hash_map<std::string, std::variant<int, std::string>> map;
  for (const auto& item : dictionary) {
    std::variant<int, std::string> value;
    try {
      value = item.second.cast<int>();
    } catch (...) {
      try {
        value = item.second.cast<std::string>();
      } catch (...) {
        continue;
      }
    }
    map.emplace(item.first.cast<std::string>(), value);
  }
  return map;
}

// Maps PT/XLA env vars to upstream torch::lazy env vars.
// Upstream lazy env vars defined in torch/csrc/lazy/core/config.h.
void MapXlaEnvVarsToLazy() {
  static bool wants_frames = xla::sys_util::GetEnvBool("XLA_IR_DEBUG", false) |
                             xla::sys_util::GetEnvBool("XLA_HLO_DEBUG", false);
  FLAGS_torch_lazy_ir_debug = wants_frames;
  static bool no_scalars =
      xla::sys_util::GetEnvBool("XLA_NO_SPECIAL_SCALARS", false);
  FLAGS_torch_lazy_handle_special_scalars = !no_scalars;
  FLAGS_torch_lazy_metrics_samples =
      xla::sys_util::GetEnvInt("XLA_METRICS_SAMPLES", 1024);
  FLAGS_torch_lazy_metrics_percentiles = xla::sys_util::GetEnvString(
      "XLA_METRICS_PERCENTILES", "0.01:0.05:0.1:0.2:0.5:0.8:0.9:0.95:0.99");
  FLAGS_torch_lazy_trim_graph_check_frequency =
      xla::sys_util::GetEnvInt("XLA_TRIM_GRAPH_CHECK_FREQUENCY", 5000);
  FLAGS_torch_lazy_trim_graph_size =
      xla::sys_util::GetEnvInt("XLA_TRIM_GRAPH_SIZE", 100000);
}

std::string GetPyTypeString(py::handle obj) {
  std::string type = obj.attr("__class__").attr("__name__").cast<std::string>();
  return type;
}

std::vector<bool> check_materialization_helper(
    const std::vector<XLATensorPtr>& xtensors) {
  std::vector<bool> need_materialization;
  need_materialization.reserve(xtensors.size());
  for (auto& xtensor : xtensors) {
    if (!xtensor) {
      // input tensor is not a xla tensor
      need_materialization.push_back(false);
    } else if (xtensor->CurrentDataHandle() != nullptr) {
      // input tensor has xla_data which means it is already on device
      need_materialization.push_back(false);
    } else if (xtensor->CurrentIrValue().node != nullptr) {
      torch::lazy::NodePtr node = xtensor->CurrentIrValue().node;
      if (torch_xla::DeviceData::Cast(xtensor->CurrentIrValue().node.get()) !=
          nullptr) {
        need_materialization.push_back(false);
      } else {
        // input tensor is an IR other than DeviceData which means a
        // compuation is required to get the value of this tensor.
        need_materialization.push_back(true);
      }
    } else {
      // TODO: maybe also handle it is a XLATensor with tensor_data case
      XLA_CHECK(false)
          << "_check_tensor_need_materialization "
             "currently does not handle XLATensor without XLAData and IR";
    }
  }
  return need_materialization;
}

void BuildProfilerSubmodule(py::module* m) {
  py::module profiler = m->def_submodule("profiler", "Profiler integration");
  py::class_<xla::profiler::ProfilerServer,
             std::unique_ptr<xla::profiler::ProfilerServer>>
      profiler_server_class(profiler, "ProfilerServer");
  profiler.def("start_server",
               [](int port) -> std::unique_ptr<xla::profiler::ProfilerServer> {
                 auto server =
                     absl::make_unique<xla::profiler::ProfilerServer>();
                 server->Start(port);
                 return server;
               },
               py::arg("port"));

  profiler.def(
      "trace",
      [](const char* service_addr, const char* logdir, int duration_ms,
         int num_tracing_attempts, int timeout_s, int interval_s,
         py::dict options) {
        absl::flat_hash_map<std::string, std::variant<int, std::string>> opts =
            ConvertDictToMap(options);
        std::chrono::seconds sleep_s(interval_s);
        tsl::Status status;
        {
          NoGilSection nogil;
          for (int i = 0; i <= timeout_s / interval_s; i++) {
            status = tensorflow::profiler::pywrap::Trace(
                service_addr, logdir, /*worker_list=*/"",
                /*include_dataset_ops=*/false, duration_ms,
                num_tracing_attempts, opts);
            if (status.ok()) {
              return;
            }
            std::this_thread::sleep_for(sleep_s);
          }
        }
        if (!status.ok()) {
          PyErr_SetString(PyExc_RuntimeError, status.error_message());
          throw py::error_already_set();
        }
      },
      py::arg("service_addr"), py::arg("logdir"), py::arg("duration_ms") = 1000,
      py::arg("num_tracing_attempts") = 3, py::arg("timeout_s") = 120,
      py::arg("interval_s") = 5, py::arg("options"));

  py::class_<xla::profiler::TraceMeWrapper> traceme_class(profiler, "TraceMe",
                                                          py::module_local());
  traceme_class.def(py::init<py::str, py::kwargs>())
      .def("__enter__", [](py::object self) -> py::object { return self; })
      .def("__exit__",
           [](py::object self, const py::object& ex_type,
              const py::object& ex_value,
              const py::object& traceback) -> py::object {
             py::cast<xla::profiler::TraceMeWrapper*>(self)->Stop();
             return py::none();
           })
      .def("set_metadata", &xla::profiler::TraceMeWrapper::SetMetadata)
      .def_static("is_enabled", &xla::profiler::TraceMeWrapper::IsEnabled);

  py::class_<torch::lazy::ScopePusher,
             std::unique_ptr<torch::lazy::ScopePusher>>
      scope_pusher_class(profiler, "ScopePusher");
  profiler.def(
      "scope_pusher",
      [](const std::string& name) -> std::unique_ptr<torch::lazy::ScopePusher> {
        return absl::make_unique<torch::lazy::ScopePusher>(name);
      });
}

void InitXlaModuleBindings(py::module m) {
  m.def("_prepare_to_exit", []() { PrepareToExit(); });
  m.def("_get_git_revs", []() { return GetRevisions(); });
  m.def("_get_xla_tensor_dimension_size",
        [](const at::Tensor& tensor, int dim) {
          return GetXlaTensorDimensionSize(tensor, dim);
        });
  m.def("_xla_nms", [](const at::Tensor& boxes, const at::Tensor& scores,
                       const at::Tensor& score_threshold,
                       const at::Tensor& iou_threshold, int64_t output_size) {
    return XlaNms(boxes, scores, score_threshold, iou_threshold, output_size);
  });
  m.def("_xla_user_computation",
        [](const std::string& opname, const std::vector<at::Tensor>& inputs,
           const ComputationPtr& computation) {
          std::vector<at::Tensor> results;
          {
            NoGilSection nogil;
            results = XlaUserComputation(opname, inputs, computation);
          }
          return results;
        });
  m.def("_get_xla_tensors_dot",
        [](const std::vector<at::Tensor>& tensors) -> std::string {
          auto coverter = [](absl::Span<const torch::lazy::Node* const> nodes) {
            return DumpUtil::ToDot(nodes);
          };
          return GetTensorsDump(tensors, coverter);
        });
  m.def("_get_xla_tensors_text",
        [](const std::vector<at::Tensor>& tensors) -> std::string {
          auto coverter = [](absl::Span<const torch::lazy::Node* const> nodes) {
            return DumpUtil::ToText(nodes);
          };
          return GetTensorsDump(tensors, coverter);
        });
  m.def("_get_xla_tensors_hlo",
        [](const std::vector<at::Tensor>& tensors) -> std::string {
          return GetTensorsHloGraph(tensors);
        });
  m.def("_xla_tensors_from_aten", [](const std::vector<at::Tensor>& tensors,
                                     const std::vector<std::string>& devices) {
    std::vector<at::Tensor> result;
    {
      NoGilSection nogil;
      std::vector<at::Tensor> xla_tensors =
          GetXlaTensorsFromAten(tensors, devices);
      result.reserve(xla_tensors.size());
      for (size_t i = 0; i < xla_tensors.size(); ++i) {
        result.push_back(torch::autograd::make_variable(
            xla_tensors[i], /*requires_grad=*/tensors.at(i).requires_grad()));
      }
    }
    return result;
  });
  m.def("_xla_get_cpu_tensors", [](const std::vector<at::Tensor>& tensors) {
    std::vector<at::Tensor> result;
    {
      NoGilSection nogil;
      std::vector<at::Tensor> cpu_tensors =
          bridge::XlaCreateTensorList(tensors);
      result.reserve(cpu_tensors.size());
      for (size_t i = 0; i < cpu_tensors.size(); ++i) {
        result.push_back(torch::autograd::make_variable(
            cpu_tensors[i], /*requires_grad=*/tensors.at(i).requires_grad()));
      }
    }
    return result;
  });
  m.def("_xla_get_tensor_view_alias_id",
        [](const at::Tensor& tensor) { return GetTensorViewAliasId(tensor); });
  m.def("_xla_get_tensor_id",
        [](const at::Tensor& tensor) { return GetTensorId(tensor); });
  m.def("_xla_get_devices",
        []() { return xla::ComputationClient::Get()->GetLocalDevices(); });
  m.def("_xla_num_devices",
        []() { return xla::ComputationClient::Get()->GetNumDevices(); });
  m.def("_xla_get_all_devices",
        []() { return xla::ComputationClient::Get()->GetAllDevices(); });
  m.def("_xla_real_devices", [](const std::vector<std::string>& devices) {
    std::vector<std::string> xla_devices;
    {
      NoGilSection nogil;
      xla_devices = GetXlaDevices(devices);
    }
    return xla_devices;
  });
  m.def("_xla_set_replication_devices",
        [](const std::vector<std::string>& devices) {
          auto replication_devices =
              std::make_shared<std::vector<std::string>>(devices);
          xla::ComputationClient::Get()->SetReplicationDevices(
              std::move(replication_devices));
        });
  m.def("_xla_get_replication_devices", []() {
    auto replication_devices =
        xla::ComputationClient::Get()->GetReplicationDevices();
    return replication_devices != nullptr ? *replication_devices
                                          : std::vector<std::string>();
  });
  m.def("_xla_get_replication_devices_count", []() {
    auto replication_devices =
        xla::ComputationClient::Get()->GetReplicationDevices();
    return replication_devices != nullptr ? replication_devices->size() : 0;
  });
  m.def("_xla_rendezvous",
        [](int ordinal, const std::string& tag, const std::string& payload,
           const std::vector<int64_t>& replicas) {
          return Rendezvous(ordinal, tag, payload, replicas);
        });

  py::class_<torch::lazy::Value, std::shared_ptr<torch::lazy::Value>>(
      m, "IrValue");
  m.def("_xla_create_token",
        [](const std::string& device) { return CreateToken(device); });
  m.def(
      "_xla_all_reduce_inplace",
      [](const std::string& reduce_type, const std::vector<at::Tensor>& tensors,
         const std::shared_ptr<torch::lazy::Value>& token, double scale,
         const py::list& groups, bool pin_layout) {
        std::vector<std::vector<int64_t>> replica_groups =
            CreateReduceGroups(groups);
        std::shared_ptr<torch::lazy::Value> new_token;
        {
          NoGilSection nogil;
          new_token = AllReduceInPlace(reduce_type, tensors, token, scale,
                                       replica_groups, pin_layout);
        }
        return new_token;
      });
  m.def("_xla_all_reduce",
        [](const std::string& reduce_type, const at::Tensor& input,
           const std::shared_ptr<torch::lazy::Value>& token, double scale,
           const py::list& groups, bool pin_layout) {
          std::vector<std::vector<int64_t>> replica_groups =
              CreateReduceGroups(groups);
          at::Tensor result;
          std::shared_ptr<torch::lazy::Value> new_token;
          {
            NoGilSection nogil;
            std::tie(result, new_token) = AllReduce(
                reduce_type, input, token, scale, replica_groups, pin_layout);
          }
          auto result_tuple = py::tuple(2);
          result_tuple[0] = torch::autograd::make_variable(
              result, /*requires_grad=*/input.requires_grad());
          result_tuple[1] = new_token;
          return result_tuple;
        });
  m.def("_xla_all_to_all",
        [](const at::Tensor& input,
           const std::shared_ptr<torch::lazy::Value>& token,
           int64_t split_dimension, int64_t concat_dimension,
           int64_t split_count, const py::list& groups, bool pin_layout) {
          std::vector<std::vector<int64_t>> replica_groups =
              CreateReduceGroups(groups);
          at::Tensor result;
          std::shared_ptr<torch::lazy::Value> new_token;
          {
            NoGilSection nogil;
            std::tie(result, new_token) =
                AllToAll(input, token, split_dimension, concat_dimension,
                         split_count, replica_groups, pin_layout);
          }
          auto result_tuple = py::tuple(2);
          result_tuple[0] = torch::autograd::make_variable(
              result, /*requires_grad=*/input.requires_grad());
          result_tuple[1] = new_token;
          return result_tuple;
        });
  m.def("_xla_all_gather", [](const at::Tensor& input,
                              const std::shared_ptr<torch::lazy::Value>& token,
                              int64_t dim, int64_t shard_count,
                              const py::list& groups, bool pin_layout) {
    std::vector<std::vector<int64_t>> replica_groups =
        CreateReduceGroups(groups);
    at::Tensor result;
    std::shared_ptr<torch::lazy::Value> new_token;
    {
      NoGilSection nogil;
      std::tie(result, new_token) =
          AllGather(input, token, dim, shard_count, replica_groups, pin_layout);
    }
    auto result_tuple = py::tuple(2);
    result_tuple[0] = torch::autograd::make_variable(
        result, /*requires_grad=*/input.requires_grad());
    result_tuple[1] = new_token;
    return result_tuple;
  });
  m.def("_xla_all_gather_out",
        [](at::Tensor& output, const at::Tensor& input,
           const std::shared_ptr<torch::lazy::Value>& token, int64_t dim,
           int64_t shard_count, const py::list& groups, bool pin_layout) {
          std::vector<std::vector<int64_t>> replica_groups =
              CreateReduceGroups(groups);
          at::Tensor result;
          std::shared_ptr<torch::lazy::Value> new_token;
          {
            NoGilSection nogil;
            new_token = AllGatherOut(output, input, token, dim, shard_count,
                                     replica_groups, pin_layout);
          }
          return new_token;
        });
  m.def("_xla_collective_permute",
        [](const at::Tensor& input,
           const std::shared_ptr<torch::lazy::Value>& token,
           const py::list& pairs) {
          std::vector<std::pair<int64_t, int64_t>> source_target_pairs =
              CreateSourceTargetPairs(pairs);
          at::Tensor result;
          std::shared_ptr<torch::lazy::Value> new_token;
          {
            NoGilSection nogil;
            std::tie(result, new_token) =
                CollectivePermute(input, token, source_target_pairs);
          }
          auto result_tuple = py::tuple(2);
          result_tuple[0] = torch::autograd::make_variable(
              result, /*requires_grad=*/input.requires_grad());
          result_tuple[1] = new_token;
          return result_tuple;
        });
  m.def("_xla_send", [](const at::Tensor& input,
                        const std::shared_ptr<torch::lazy::Value>& token,
                        int64_t channel_id) {
    // The input will be returned as result.
    at::Tensor input_as_result;
    std::shared_ptr<torch::lazy::Value> new_token;
    {
      NoGilSection nogil;
      std::tie(input_as_result, new_token) = Send(input, token, channel_id);
    }
    auto result_tuple = py::tuple(2);
    result_tuple[0] = torch::autograd::make_variable(input_as_result,
                                                     /*requires_grad=*/false);
    result_tuple[1] = new_token;
    return result_tuple;
  });
  m.def("_xla_recv",
        [](at::Tensor& output, const std::shared_ptr<torch::lazy::Value>& token,
           int64_t channel_id) {
          at::Tensor result;
          std::shared_ptr<torch::lazy::Value> new_token;
          {
            NoGilSection nogil;
            std::tie(result, new_token) = Recv(output, token, channel_id);
          }
          auto result_tuple = py::tuple(2);
          result_tuple[0] = torch::autograd::make_variable(
              result, /*requires_grad=*/output.requires_grad());
          result_tuple[1] = new_token;
          return result_tuple;
        });
  m.def("_xla_reduce_scatter",
        [](const std::string& reduce_type, const at::Tensor& input,
           const std::shared_ptr<torch::lazy::Value>& token, double scale,
           int64_t scatter_dim, int64_t shard_count, const py::list& groups,
           bool pin_layout) {
          std::vector<std::vector<int64_t>> replica_groups =
              CreateReduceGroups(groups);
          at::Tensor result;
          std::shared_ptr<torch::lazy::Value> new_token;
          {
            NoGilSection nogil;
            std::tie(result, new_token) =
                ReduceScatter(reduce_type, input, token, scale, scatter_dim,
                              shard_count, replica_groups, pin_layout);
          }
          auto result_tuple = py::tuple(2);
          result_tuple[0] = torch::autograd::make_variable(
              result, /*requires_grad=*/input.requires_grad());
          result_tuple[1] = new_token;
          return result_tuple;
        });
  m.def("_xla_reduce_scatter_out",
        [](const std::string& reduce_type, at::Tensor& output,
           const at::Tensor& input,
           const std::shared_ptr<torch::lazy::Value>& token, double scale,
           int64_t scatter_dim, int64_t shard_count, const py::list& groups,
           bool pin_layout) {
          std::vector<std::vector<int64_t>> replica_groups =
              CreateReduceGroups(groups);
          at::Tensor result;
          std::shared_ptr<torch::lazy::Value> new_token;
          {
            NoGilSection nogil;
            new_token = ReduceScatterOut(reduce_type, output, input, token,
                                         scale, scatter_dim, shard_count,
                                         replica_groups, pin_layout);
          }
          return new_token;
        });
  m.def("_xla_optimization_barrier_",
        [](std::vector<at::Tensor>& inputs) { OptimizationBarrier_(inputs); });
  m.def("_xla_set_default_device", [](const std::string& device) {
    return SetCurrentThreadDevice(device);
  });
  m.def("_xla_get_default_device", []() { return GetCurrentThreadDevice(); });
  m.def("_xla_get_default_device_ordinal", []() {
    std::string device_str = GetCurrentThreadDevice();
    torch::lazy::BackendDevice device =
        bridge::AtenDeviceToXlaDevice(device_str);
    return device.ordinal();
  });
  m.def("_xla_get_process_index",
        []() { return xla::ComputationClient::Get()->GetProcessIndex(); });
  m.def("_xla_get_num_processes",
        []() { return xla::ComputationClient::Get()->GetNumProcesses(); });
  m.def("_xla_get_device_ordinal", [](const std::string& device_str) {
    return bridge::AtenDeviceToXlaDevice(device_str).ordinal();
  });
  m.def("_xla_get_device_attributes", [](const std::string& device_str) {
    const absl::flat_hash_map<
        std::string, xla::ComputationClient::DeviceAttribute>& attributes =
        xla::ComputationClient::Get()->GetDeviceAttributes(
            bridge::AtenDeviceToXlaDevice(device_str).toString());

    py::dict dict;
    for (auto const& [name, value] : attributes) {
      dict[py::str(name)] = py::cast(value);
    }
    return dict;
  });
  m.def("_xla_set_rng_seed",
        [](uint64_t seed, const std::string& device) {
          SetRngSeed(seed, device);
        },
        py::arg("seed") = 101, py::arg("device") = "");
  m.def("_xla_get_rng_seed",
        [](const std::string& device) { return GetRngSeed(device); },
        py::arg("device") = "");
  m.def("_xla_sync_multi",
        [](const std::vector<at::Tensor>& tensors,
           const std::vector<std::string>& devices, bool wait,
           bool sync_xla_data) {
          NoGilSection nogil;
          SyncTensors(tensors, devices, wait, sync_xla_data);
        },
        py::arg("tensors"), py::arg("devices"), py::arg("wait") = true,
        py::arg("sync_xla_data") = true);
  m.def("_xla_warm_up_cache",
        [](const std::vector<at::Tensor>& tensors,
           const std::vector<std::string>& devices) {
          NoGilSection nogil;
          SyncTensors(tensors, devices, /*wait=*/false, /*sync_xla_data=*/false,
                      /*warm_up_cache_only=*/true);
        },
        py::arg("tensors"), py::arg("devices"));
  m.def("_xla_sync_live_tensors",
        [](const std::string& device, const std::vector<std::string>& devices,
           bool wait) {
          NoGilSection nogil;
          SyncLiveTensors(device, devices, wait);
        },
        py::arg("device") = "", py::arg("devices"), py::arg("wait") = true);
  m.def("_xla_step_marker",
        [](const std::string& device, const std::vector<std::string>& devices,
           bool wait) {
          NoGilSection nogil;
          StepMarker(device, devices, wait);
        },
        py::arg("device") = "", py::arg("devices"), py::arg("wait") = true);
  m.def("_xla_wait_device_ops",
        [](const std::vector<std::string>& devices) {
          NoGilSection nogil;
          XLAGraphExecutor::Get()->WaitDeviceOps(devices);
          xla::ComputationClient::Get()->WaitDeviceOps(devices);
        },
        py::arg("devices"));
  m.def("_xla_counter_names", []() {
    auto counter_names = torch::lazy::GetCounterNames();
    auto xla_counter_names = xla::metrics::GetCounterNames();
    counter_names.insert(counter_names.end(), xla_counter_names.begin(),
                         xla_counter_names.end());
    return counter_names;
  });
  m.def("_xla_counter_value", [](const std::string& name) -> py::object {
    auto* data = torch::lazy::GetCounter(name);
    if (data != nullptr) {
      return py::cast<int64_t>(data->Value());
    }

    auto* xla_data = xla::metrics::GetCounter(name);
    return xla_data != nullptr ? py::cast<int64_t>(xla_data->Value())
                               : py::none();
  });
  m.def("_xla_metric_names", []() {
    auto metric_names = torch::lazy::GetMetricNames();
    auto xla_metric_names = xla::metrics::GetMetricNames();
    metric_names.insert(metric_names.end(), xla_metric_names.begin(),
                        xla_metric_names.end());
    return metric_names;
  });
  m.def("_xla_metric_data", [](const std::string& name) -> py::object {
    return GetMetricData(name);
  });
  m.def("_xla_metrics_report", []() {
    // NOTE: [TORCH_LAZY_COUNTER v.s. XLA_COUNTER]
    // Counters and Metrics are divided into two groups: one in PyTorch/XLA and
    // another in ComputationClient. Therefore, we need to stitch the report
    // together. Ideally, those two sets shouldn't have any overlaps. The reason
    // why is that we cannot have ComputationClient to use the
    // TORCH_LAZY_COUNTER as it currently cannot depend on PyTorch (as part of
    // TensorFlow).
    // TODO(jwtan): Unify them once ComputationClient becomes a standalone
    // library.
    return torch::lazy::CreateMetricReport() +
           xla::metrics_reader::CreateMetricReport(
               xla::ComputationClient::Get()->GetMetrics());
  });
  m.def("_short_xla_metrics_report", [](const py::list& counter_names,
                                        const py::list& metric_names) {
    std::vector<std::string> counter_name_vec;
    std::vector<std::string> metric_name_vec;
    for (auto& counter : counter_names) {
      counter_name_vec.push_back(counter.cast<std::string>());
    }
    for (auto& metric : metric_names) {
      metric_name_vec.push_back(metric.cast<std::string>());
    }
    // See NOTE: [TORCH_LAZY_COUNTER v.s. XLA_COUNTER].
    return torch::lazy::CreateMetricReport(counter_name_vec, metric_name_vec) +
           xla::metrics_reader::CreateMetricReport(counter_name_vec,
                                                   metric_name_vec);
  });
  m.def("_clear_xla_counters", []() {
    torch::lazy::MetricsArena::Get()->ResetCounters();
    xla::metrics::ClearCounters();
  });
  m.def("_clear_xla_metrics", []() {
    torch::lazy::MetricsArena::Get()->ResetMetrics();
    xla::metrics::ClearMetrics();
  });
  m.def("_xla_tensors_report",
        [](size_t nodes_threshold, const std::string& device) {
          return GetLiveTensorsReport(nodes_threshold, device);
        },
        py::arg("nodes_threshold") = 100, py::arg("device") = "");
  m.def("_xla_memory_info", [](const std::string& device) -> py::object {
    return GetMemoryInfo(device);
  });
  m.def("_xla_set_use_full_mat_mul_precision",
        [](bool use_full_mat_mul_precision) {
          XlaHelpers::set_mat_mul_precision(
              use_full_mat_mul_precision ? xla::PrecisionConfig::HIGHEST
                                         : xla::PrecisionConfig::DEFAULT);
        },
        py::arg("use_full_mat_mul_precision") = true);

  py::class_<xla::util::RecordReader, std::shared_ptr<xla::util::RecordReader>>(
      m, "RecordReader");
  m.def("_xla_create_tfrecord_reader",
        [](const std::string& path, const std::string& compression,
           int64_t buffer_size) {
          NoGilSection nogil;
          return CreateRecordReader(path, compression, buffer_size);
        },
        py::arg("path"), py::arg("compression") = "",
        py::arg("buffer_size") = 16 * 1024 * 1024);
  m.def(
      "_xla_tfrecord_read",
      [](const std::shared_ptr<xla::util::RecordReader>& reader) -> py::object {
        xla::util::RecordReader::Data record;
        if (!RecordRead(reader, &record)) {
          return py::none();
        }
        return py::bytes(record.data(), record.size());
      });
  m.def("_xla_tfexample_read",
        [](const std::shared_ptr<xla::util::RecordReader>& reader) {
          return RecordReadExample(reader);
        });

  py::class_<tsl::RandomAccessFile>(m, "TfRdFile");
  m.def("_xla_tffile_open", [](const std::string& path) {
    std::unique_ptr<tsl::RandomAccessFile> file;
    {
      NoGilSection nogil;
      file = OpenTfFile(path);
    }
    return py::cast(file.release(),
                    pybind11::return_value_policy::take_ownership);
  });
  m.def("_xla_tffile_stat",
        [](const std::string& path) { return StatTfFile(path); });
  m.def("_xla_tffile_read",
        [](tsl::RandomAccessFile* file, uint64_t offset, size_t size) {
          return ReadTfFile(file, offset, size);
        });

  py::class_<tsl::WritableFile>(m, "TfWrFile");
  m.def("_xla_tffile_create", [](const std::string& path) {
    std::unique_ptr<tsl::WritableFile> file;
    {
      NoGilSection nogil;
      file = CreateTfFile(path);
    }
    return py::cast(file.release(),
                    pybind11::return_value_policy::take_ownership);
  });
  m.def("_xla_tffile_write",
        [](tsl::WritableFile* file, const std::string& data) {
          NoGilSection nogil;
          WriteTfFile(file, data);
        });
  m.def("_xla_tffile_flush", [](tsl::WritableFile* file) {
    NoGilSection nogil;
    FlushTfFile(file);
  });

  m.def("_xla_tffs_list",
        [](const std::string& pattern) { return ListTfFs(pattern); });
  m.def("_xla_tffs_remove", [](const std::string& path) {
    NoGilSection nogil;
    RemoveTfFile(path);
  });

  py::class_<xla::XlaBuilder, op_builder::BuilderPtr>(m, "XlaBuilder");
  py::class_<op_builder::Op, op_builder::OpPtr>(m, "XlaOp");
  py::class_<Computation, ComputationPtr>(m, "XlaComputation");
  m.def("_xla_op_create_builder", [](const std::string& name) {
    return std::make_shared<xla::XlaBuilder>(name);
  });
  m.def("_xla_op_tensor_shape",
        [](const at::Tensor& tensor, const std::string& device) {
          xla::Shape tensor_shape = GetTensorShape(tensor, device);
          return op_builder::ShapeToPyShape(tensor_shape);
        });
  m.def("_xla_op_param", [](op_builder::BuilderPtr builder, int64_t param_no,
                            py::object py_shape) {
    xla::Shape shape = op_builder::PyShapeToShape(py_shape);
    xla::XlaOp param = xla::Parameter(builder.get(), param_no, shape,
                                      absl::StrCat("p", param_no));
    return std::make_shared<op_builder::Op>(std::move(builder),
                                            std::move(param));
  });
  m.def("_xla_op_build", [](const std::string& name, op_builder::OpPtr root) {
    ComputationPtr computation;
    {
      NoGilSection nogil;
      computation = CreateComputation(name, root->op);
    }
    return computation;
  });
  m.def("_xla_op_computation_from_module_proto",
        [](const std::string& name, const std::string& module_proto) {
          ComputationPtr computation;
          {
            NoGilSection nogil;
            computation = CreateComputationFromProto(name, module_proto);
          }
          return computation;
        });
  m.def("_xla_computation_text", [](const ComputationPtr& computation) {
    std::string hlo_text;
    {
      NoGilSection nogil;
      hlo_text = ConsumeValue(
          xla::util::GetComputationHloText(computation->computation()));
    }
    return hlo_text;
  });
  m.def("_xla_op_shape", [](op_builder::OpPtr op) {
    const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(op->op);
    return op_builder::ShapeToPyShape(shape);
  });
  m.def("_xla_op_builder", [](op_builder::OpPtr op) { return op->builder; });
  m.def("_xla_op_create",
        [](op_builder::BuilderPtr builder, const std::string& opname,
           const std::vector<op_builder::OpPtr>& operands, py::dict args) {
          return op_builder::CreateOp(builder, opname, operands, args);
        });
  m.def("_run_xrt_local_service", [](uint64_t service_port) {
    xla::ComputationClient::RunLocalService(service_port);
  });
  m.def("_xla_sgd_optimizer_step_",
        [](const at::Tensor& found_inf, at::Tensor& step, at::Tensor& param,
           at::Tensor& buf, const at::Tensor& d_p, double weight_decay,
           double momentum, double lr, double dampening, bool nesterov,
           bool maximize) {
          {
            NoGilSection nogil;
            XLATensorPtr found_inf_xla = bridge::GetXlaTensor(found_inf);
            XLATensorPtr step_xla = bridge::GetXlaTensor(step);
            XLATensorPtr param_xla = bridge::GetXlaTensor(param);
            XLATensorPtr d_p_xla = bridge::GetXlaTensor(d_p);
            XLATensorPtr buf_xla = bridge::GetXlaTensor(buf);
            tensor_methods::sgd_optimizer_step_(
                found_inf_xla, step_xla, param_xla, buf_xla, d_p_xla,
                weight_decay, momentum, lr, dampening, nesterov, maximize);
          }
        });
  m.def("_xla_adam_optimizer_step_",
        [](const at::Tensor& found_inf, at::Tensor& step, at::Tensor& param,
           at::Tensor& grad, at::Tensor& exp_avg, at::Tensor& exp_avg_sq,
           at::Tensor& max_exp_avg_sq, double beta1, double beta2, double lr,
           double weight_decay, double eps, bool amsgrad, bool maximize,
           bool use_adamw) {
          {
            NoGilSection nogil;
            XLATensorPtr found_inf_xla = bridge::GetXlaTensor(found_inf);
            XLATensorPtr step_xla = bridge::GetXlaTensor(step);
            XLATensorPtr param_xla = bridge::GetXlaTensor(param);
            XLATensorPtr grad_xla = bridge::GetXlaTensor(grad);
            XLATensorPtr exp_avg_xla = bridge::GetXlaTensor(exp_avg);
            XLATensorPtr exp_avg_sq_xla = bridge::GetXlaTensor(exp_avg_sq);
            XLATensorPtr max_exp_avg_sq_xla =
                bridge::GetXlaTensor(max_exp_avg_sq);
            tensor_methods::adam_optimizer_step_(
                found_inf_xla, step_xla, param_xla, grad_xla, exp_avg_xla,
                exp_avg_sq_xla, max_exp_avg_sq_xla, beta1, beta2, lr,
                weight_decay, eps, amsgrad, maximize, use_adamw);
          }
        });
  m.def("_xla_mark_sharding", [](const at::Tensor& input,
                                 const py::list& tile_assignment,
                                 bool replicated = false, bool manual = false) {
    TORCH_LAZY_COUNTER("XlaMarkSharding", 1);
    XLATensorPtr xtensor = bridge::GetXlaTensor(input);
    xla::OpSharding sharding =
        ShardingUtil::CreateOpSharding(tile_assignment, replicated, manual);
    auto new_sharding_spec = std::make_shared<XLATensor::ShardingSpec>(
        sharding, MakeShapeWithDeviceLayout(
                      xtensor->shape(),
                      static_cast<XlaDeviceType>(xtensor->GetDevice().type())));

    at::Tensor cpu_tensor;
    if (xla::sys_util::GetEnvBool("XLA_USE_SPMD", false) &&
        xtensor->CurrentTensorData().has_value()) {
      TORCH_LAZY_COUNTER("VirtualDeviceUsage", 1);
      // When virtual device is enabled for SPMD, we defer the initial data
      // transfer to the device and retain the original data on the host, until
      // the sharded data transfer.
      cpu_tensor = xtensor->CurrentTensorData().value();
    } else {
      // A new input tensor is not expected to be sharded. But sometimes, the
      // same input is used sharding annotation, in which case we can skip if
      // it's the same sharding; however, if it's the same input with a
      // different sharding then we block & ask the user to clear the existing
      // sharding first.
      auto current_sharding_spec = xtensor->sharding_spec();
      if (current_sharding_spec) {
        XLA_CHECK(ShardingUtil::EqualShardingSpecs(*new_sharding_spec,
                                                   *current_sharding_spec))
            << "Existing annotation must be cleared first.";
        return;
      }

      // If the at::Tensor data is not present, we need to re-download the
      // tensor from the physical device to CPU. In that case, the value
      // must be present on the backend device.
      XLA_CHECK(xtensor->GetXlaData() != nullptr &&
                xtensor->CurrentDataHandle()->HasValue())
          << "Cannot shard tensor. Data not present on any device.";
      std::vector<XLATensorPtr> xla_tensors{xtensor};
      cpu_tensor = XLAGraphExecutor::Get()->GetTensors(&xla_tensors)[0];
    }
    auto xla_data = CreateTensorsData(
        std::vector<at::Tensor>{cpu_tensor},
        std::vector<XLATensor::ShardingSpecPtr>{new_sharding_spec},
        std::vector<std::string>{GetVirtualDevice().toString()})[0];
    xtensor->SetXlaData(xla_data);
    xtensor->SetShardingSpec(*new_sharding_spec);

    // Register sharded tensor data.
    XLAGraphExecutor::Get()->RegisterTensor(xtensor->data());
  });
  m.def("_xla_clear_sharding", [](const at::Tensor& input) {
    XLATensorPtr xtensor = bridge::GetXlaTensor(input);
    xtensor->ClearShardingSpec();
  });
  m.def("_get_xla_sharding_spec", [](const at::Tensor& input) -> std::string {
    XLATensorPtr xtensor = bridge::GetXlaTensor(input);
    auto sharding_spec = xtensor->sharding_spec();
    if (sharding_spec != nullptr) {
      auto hlo_sharding = xla::HloSharding::FromProto(sharding_spec->sharding);
      return hlo_sharding->ToString();
    }
    return std::string();
  });
  // This is useful for debugging and generating a partitioned HLO separately
  // outside the actual compilation & execution. This allows testing with
  // different partitioning configurations.
  m.def("_xla_partitioning_pass",
        [](const std::vector<at::Tensor>& tensors, int64_t num_replicas,
           int64_t num_devices, bool conv_halo_exchange_always_on_lhs = true,
           bool choose_faster_windowed_einsum = false,
           bool unroll_windowed_einsum = false,
           bool bidirectional_windowed_einsum = false) -> std::string {
          xla::HloModuleConfig config;
          config.set_use_spmd_partitioning(true);
          config.set_replica_count(num_replicas);
          config.set_num_partitions(num_devices);

          std::string hlo_text = GetTensorsHloGraph(tensors);
          auto hlo_module_error =
              xla::ParseAndReturnUnverifiedModule(hlo_text, config);
          XLA_CHECK_OK(hlo_module_error.status())
              << "HLO Module loading failed: " << hlo_module_error.status();

          auto module = std::move(hlo_module_error.value());
          xla::HloModuleProto module_proto = ShardingUtil::SpmdPartitioningPass(
              module->ToProto(), num_replicas, num_devices,
              conv_halo_exchange_always_on_lhs, choose_faster_windowed_einsum,
              unroll_windowed_einsum, bidirectional_windowed_einsum);
          module = std::move(
              xla::HloModule::CreateFromProto(module_proto, config).value());
          return module->ToString();
        });
  m.def("_init_xla_lazy_backend", []() {
    MapXlaEnvVarsToLazy();
    InitXlaBackend();
  });
  m.def("_replace_xla_tensor",
        [](at::Tensor& self, const at::Tensor& source) -> at::Tensor& {
          return XLANativeFunctions::set_(self, source);
        });

  /* The distributed runtime service is used by the PjRt GPU client. */
  py::class_<xla::DistributedRuntimeService,
             std::unique_ptr<xla::DistributedRuntimeService>>
      distributed_runtime_service(m, "DistributedRuntimeService");
  distributed_runtime_service.def("shutdown",
                                  &xla::DistributedRuntimeService::Shutdown,
                                  py::call_guard<py::gil_scoped_release>());
  m.def("_xla_get_distributed_runtime_service",
        [](int num_nodes) -> std::unique_ptr<xla::DistributedRuntimeService> {
          std::string dist_service_addr =
              xla::sys_util::GetEnvString("PJRT_DIST_SERVICE_ADDR", "");
          XLA_CHECK(!dist_service_addr.empty())
              << "Must set PJRT_DIST_SERVICE_ADDR environment variable to use "
                 "distributed runtime";
          XLA_CHECK(num_nodes > 0)
              << "num_nodes must be positive: " << num_nodes;

          xla::DistributedRuntimeServiceImpl::Options options;
          options.num_nodes = num_nodes;
          return std::move(xla::GetDistributedRuntimeService(
                               dist_service_addr, options,
                               /*use_coordination_service=*/false)
                               .value());
        });

  BuildProfilerSubmodule(&m);

  m.def("_get_tensors_handle",
        [](const std::vector<at::Tensor>& tensors) -> std::vector<int64_t> {
          std::vector<int64_t> handles;
          handles.reserve(tensors.size());
          for (auto& tensor : tensors) {
            handles.push_back(bridge::GetXlaTensor(tensor)->GetOpaqueHandle());
          }
          return handles;
        });

  // -------------Dynamo Integration API Start-------------------------
  /*
   * Return tensor ids and tensors for DeviceData nodes.
   */
  m.def("_get_tensors_xla_device_data_node",
        [](const std::vector<at::Tensor>& tensors)
            -> std::pair<std::vector<int64_t>, std::vector<at::IValue>> {
          std::vector<int64_t> tensor_ids;
          std::vector<at::IValue> ivalues;
          std::vector<const torch::lazy::Node*> roots;
          for (const at::Tensor& tensor : tensors) {
            auto xtensor = bridge::TryGetXlaTensor(tensor);
            if (xtensor) {
              roots.push_back(xtensor->GetIrValue().node.get());
            }
          }
          auto post_order = torch::lazy::Util::ComputePostOrder(roots);
          std::unordered_set<xla::ComputationClient::Data::OpaqueHandle>
              data_handles;

          for (const torch::lazy::Node* nodeptr : post_order) {
            const auto backend_data =
                torch::lazy::getBackend()->GetComputationDataFromNode(nodeptr);
            if (!backend_data) {
              continue;
            }

            // Dedup by handle
            xla::ComputationClient::Data::OpaqueHandle handle =
                backend_data->GetHandle();
            if (!data_handles.insert(handle).second) {
              continue;
            }
            auto* infoptr =
                static_cast<torch::lazy::LazyGraphExecutor::DeviceDataInfo*>(
                    backend_data->info());
            if (infoptr) {
              tensor_ids.push_back(infoptr->tensor_id);
            } else {
              // TODO(JackCaoG): Make sure this device data is actually seed.
              tensor_ids.push_back(seed_info_id);
            }
            at::Tensor tensor = bridge::AtenFromXlaTensor(
                torch_xla::XLATensor::Create(backend_data));
            ivalues.emplace_back(tensor);
          }
          return std::make_pair(tensor_ids, ivalues);
        });

  m.def("_get_seed_info_id", []() -> int64_t { return seed_info_id; });

  m.def("_get_base_seed_as_tensor",
        [](const std::string& device_str) -> at::IValue {
          torch::lazy::BackendDevice device =
              bridge::AtenDeviceToXlaDevice(c10::Device(device_str));
          return bridge::AtenFromXlaTensor(torch_xla::XLATensor::Create(
              XLAGraphExecutor::Get()->GetBaseSeedData(device)));
        });

  // Return true if value of the tensor requires a computation.
  m.def("_check_tensor_need_materialization",
        [](const std::vector<at::Tensor>& tensors) -> std::vector<bool> {
          std::vector<XLATensorPtr> xtensors;
          xtensors.reserve(tensors.size());
          for (const at::Tensor& tensor : tensors) {
            xtensors.push_back(bridge::TryGetXlaTensor(tensor));
          }
          return check_materialization_helper(xtensors);
        });

  // Return true if value of the any tensor in this devicerequires a
  // computation.
  m.def("_check_device_tensor_need_materialization",
        [](const std::string& device_str) -> std::vector<bool> {
          auto opt_device = GetOptionalDevice(device_str);
          std::vector<XLATensorPtr> xtensors =
              XLAGraphExecutor::Get()->GetLiveTensors(
                  opt_device ? &opt_device.value() : nullptr);
          return check_materialization_helper(xtensors);
        });

  m.def("_get_graph_hash", [](const std::vector<at::Tensor>& tensors) {
    std::vector<XLATensorPtr> xtensors;
    xtensors.reserve(tensors.size());
    for (auto& tensor : tensors) {
      xtensors.push_back(bridge::GetXlaTensor(tensor));
    }
    torch::lazy::hash_t hash = XLAGraphExecutor::Get()->GetGraphHash(xtensors);
    std::string bin((const char*)&hash, sizeof(hash));
    return py::bytes(bin);
  });

  m.def("_clear_pending_irs", [](const std::string& device) {
    // Use with caution. Those tensor whole ir was cleared with be replaced
    // with a placeholder XLAData and SHOULD NOT be accessed.
    ClearPendingIrs(device);
  });

  m.def("_run_cached_graph",
        [](const std::string& hash_str,
           const std::vector<at::IValue>& graph_inputs)
            -> std::vector<at::Tensor> {
          XLA_CHECK(hash_str.size() == sizeof(torch::lazy::hash_t));
          torch::lazy::hash_t hash = *(torch::lazy::hash_t*)(hash_str.c_str());
          torch::lazy::BackendDevice device = torch_xla::GetCurrentDevice();
          auto results = XLAGraphExecutor::Get()->ExecuteComputationWithBarrier(
              hash, graph_inputs, device);
          std::vector<at::Tensor> retlist;
          {
            TORCH_LAZY_TIMED("RunCachedGraphOutputData");
            // Convert result back to at::tensor
            int i = 0;
            for (auto& data : results) {
              XLATensorPtr xla_tensor = torch_xla::XLATensor::Create(data);
              retlist.push_back(bridge::AtenFromXlaTensor(xla_tensor));
              ++i;
            }
          }

          return retlist;
        });
  // -------------Dynamo Integration API End-------------------------
}
}  // namespace

void InitXlaBindings(py::module m) { InitXlaModuleBindings(m); }

}  // namespace torch_xla

PYBIND11_MODULE(_XLAC, m) { torch_xla::InitXlaBindings(m); }
