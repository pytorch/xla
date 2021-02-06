#include <Python.h>
#include <c10/core/Device.h>
#include <c10/util/Optional.h>

#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/types/variant.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/mesh_service.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/metrics_analysis.h"
#include "tensorflow/compiler/xla/xla_client/metrics_reader.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/profiler.h"
#include "tensorflow/compiler/xla/xla_client/record_reader.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/python/profiler/internal/profiler_pywrap_impl.h"
#include "tensorflow/python/profiler/internal/traceme_wrapper.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/python/pybind.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/aten_xla_type.h"
#include "torch_xla/csrc/computation.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/ir_util.h"
#include "torch_xla/csrc/python_util.h"
#include "torch_xla/csrc/tensor_impl.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/version.h"
#include "torch_xla/csrc/xla_op_builder.h"

namespace torch_xla {
namespace {

struct NoGilSection {
  NoGilSection() : state(PyEval_SaveThread()) {}
  ~NoGilSection() { PyEval_RestoreThread(state); }
  PyThreadState* state = nullptr;
};

c10::optional<Device> GetOptionalDevice(const std::string& device_str) {
  if (device_str.empty()) {
    return c10::nullopt;
  }
  return bridge::AtenDeviceToXlaDevice(c10::Device(device_str));
}

Device GetDeviceOrCurrent(const std::string& device_str) {
  if (device_str.empty()) {
    return GetCurrentDevice();
  }
  return bridge::AtenDeviceToXlaDevice(c10::Device(device_str));
}

void PrepareToExit() {
  xla::ComputationClient* client = xla::ComputationClient::GetIfInitialized();
  if (client != nullptr) {
    XLATensor::WaitDeviceOps({});
    client->PrepareToExit();
  }
}

std::string GetTensorsDump(
    const std::vector<at::Tensor>& tensors,
    const std::function<std::string(absl::Span<const ir::Node* const>)>&
        coverter) {
  std::vector<const ir::Node*> nodes;
  std::vector<ir::Value> values;
  for (auto& tensor : tensors) {
    XLATensor xtensor = bridge::GetXlaTensor(tensor);
    values.push_back(xtensor.GetIrValue());
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
    Device device = bridge::AtenDeviceToXlaDevice(c10::Device(device_str));
    xla_devices.emplace_back(device.ToString());
  }
  return xla_devices;
}

std::vector<XLATensor> GetXlaTensors(const std::vector<at::Tensor>& tensors,
                                     bool want_all) {
  std::vector<XLATensor> xtensors;
  xtensors.reserve(tensors.size());
  if (want_all) {
    for (auto& tensor : tensors) {
      xtensors.push_back(bridge::GetXlaTensor(tensor));
    }
  } else {
    for (auto& tensor : tensors) {
      auto xtensor = bridge::TryGetXlaTensor(tensor);
      if (xtensor) {
        xtensors.push_back(*xtensor);
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

std::vector<std::vector<xla::int64>> CreateReduceGroups(
    const py::list& groups) {
  std::vector<std::vector<xla::int64>> replica_groups;
  for (auto& group : groups) {
    replica_groups.emplace_back();
    for (auto& replica_id : group.cast<py::list>()) {
      replica_groups.back().push_back(replica_id.cast<xla::int64>());
    }
  }
  return replica_groups;
}

std::vector<std::pair<xla::int64, xla::int64>> CreateSourceTargetPairs(
    const py::list& pairs) {
  std::vector<std::pair<xla::int64, xla::int64>> source_target_pairs;
  for (auto& pair : pairs) {
    const auto& pylist_pair = pair.cast<py::list>();
    XLA_CHECK_EQ(len(pylist_pair), 2);
    source_target_pairs.push_back(
        {pylist_pair[0].cast<xla::int64>(), pylist_pair[1].cast<xla::int64>()});
  }
  return source_target_pairs;
}

std::shared_ptr<ir::Value> AllReduceInPlace(
    const std::string& reduce_type, const std::vector<at::Tensor>& tensors,
    const std::shared_ptr<ir::Value>& token, double scale,
    const std::vector<std::vector<xla::int64>>& replica_groups) {
  std::vector<XLATensor> xtensors = GetXlaTensors(tensors, /*want_all=*/true);
  return std::make_shared<ir::Value>(XLATensor::all_reduce(
      &xtensors, *token, GetReduceType(reduce_type), scale, replica_groups));
}

std::pair<at::Tensor, std::shared_ptr<ir::Value>> AllReduce(
    const std::string& reduce_type, const at::Tensor& input,
    const std::shared_ptr<ir::Value>& token, double scale,
    const std::vector<std::vector<xla::int64>>& replica_groups) {
  XLATensor result;
  ir::Value new_token;
  std::tie(result, new_token) =
      XLATensor::all_reduce(bridge::GetXlaTensor(input), *token,
                            GetReduceType(reduce_type), scale, replica_groups);
  return std::pair<at::Tensor, std::shared_ptr<ir::Value>>(
      bridge::AtenFromXlaTensor(std::move(result)),
      std::make_shared<ir::Value>(new_token));
}

std::pair<at::Tensor, std::shared_ptr<ir::Value>> AllToAll(
    const at::Tensor& input, const std::shared_ptr<ir::Value>& token,
    xla::int64 split_dimension, xla::int64 concat_dimension,
    xla::int64 split_count,
    const std::vector<std::vector<xla::int64>>& replica_groups) {
  XLATensor result;
  ir::Value new_token;
  std::tie(result, new_token) = XLATensor::all_to_all(
      bridge::GetXlaTensor(input), *token, split_dimension, concat_dimension,
      split_count, replica_groups);
  return std::pair<at::Tensor, std::shared_ptr<ir::Value>>(
      bridge::AtenFromXlaTensor(std::move(result)),
      std::make_shared<ir::Value>(new_token));
}

std::pair<at::Tensor, std::shared_ptr<ir::Value>> CollectivePermute(
    const at::Tensor& input, const std::shared_ptr<ir::Value>& token,
    const std::vector<std::pair<xla::int64, xla::int64>>& source_target_pairs) {
  XLATensor result;
  ir::Value new_token;
  std::tie(result, new_token) = XLATensor::collective_permute(
      bridge::GetXlaTensor(input), *token, source_target_pairs);
  return std::pair<at::Tensor, std::shared_ptr<ir::Value>>(
      bridge::AtenFromXlaTensor(std::move(result)),
      std::make_shared<ir::Value>(new_token));
}

void SyncTensors(const std::vector<at::Tensor>& tensors,
                 const std::vector<std::string>& devices, bool wait,
                 bool sync_xla_data) {
  std::vector<XLATensor> xtensors = GetXlaTensors(tensors, /*want_all=*/false);
  XLATensor::SyncTensorsGraph(&xtensors, devices, wait, sync_xla_data);
}

void SyncLiveTensors(const std::string& device_str,
                     const std::vector<std::string>& devices, bool wait) {
  auto opt_device = GetOptionalDevice(device_str);
  XLATensor::SyncLiveTensorsGraph(opt_device ? &opt_device.value() : nullptr,
                                  devices, wait);
}

void StepMarker(const std::string& device_str,
                const std::vector<std::string>& devices, bool wait) {
  tensorflow::profiler::TraceMe activity(
      "StepMarker", tensorflow::profiler::TraceMeLevel::kInfo);
  Device device = GetDeviceOrCurrent(device_str);
  XLATensor::SyncLiveTensorsGraph(&device, devices, wait);
  XLATensor::MarkStep(device);
  bool debug_mode = xla::sys_util::GetEnvBool("PT_XLA_DEBUG", false);
  if (TF_PREDICT_FALSE(debug_mode)) {
    std::string report = xla::metrics::CreatePerformanceReport();
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

void SetRngSeed(xla::uint64 seed, const std::string& device_str) {
  Device device = GetDeviceOrCurrent(device_str);
  XLATensor::SetRngSeed(device, seed);
}

xla::uint64 GetRngSeed(const std::string& device_str) {
  return XLATensor::GetRunningSeed(GetDeviceOrCurrent(device_str));
}

std::string GetTensorsHloGraph(const std::vector<at::Tensor>& tensors) {
  std::vector<XLATensor> xtensors = GetXlaTensors(tensors, /*want_all=*/false);
  return XLATensor::DumpHloComputation(xtensors);
}

std::string GetLiveTensorsReport(size_t nodes_threshold,
                                 const std::string& device_str) {
  auto opt_device = GetOptionalDevice(device_str);
  auto tensors =
      XLATensor::GetLiveTensors(opt_device ? &opt_device.value() : nullptr);
  std::stringstream ss;
  for (auto& tensor : tensors) {
    ir::Value ir_value = tensor.CurrentIrValue();
    if (ir_value) {
      std::vector<const ir::Node*> roots({ir_value.node.get()});
      auto post_order = ir::Util::ComputePostOrder(roots);
      if (post_order.size() > nodes_threshold) {
        ss << "Tensor: id=" << tensor.GetUniqueId()
           << ", shape=" << tensor.shape().get()
           << ", device=" << tensor.GetDevice()
           << ", ir_nodes=" << post_order.size() << "\n";
        for (size_t i = post_order.size(); i > 0; --i) {
          if (!post_order[i - 1]->metadata().frame_info.empty()) {
            ss << post_order[i - 1]->metadata().frame_info;
            break;
          }
        }
        ss << ir::DumpUtil::PostOrderToText(post_order, roots);
        ss << "\n\n";
      }
    }
  }
  return ss.str();
}

std::ptrdiff_t GetTensorViewAliasId(const at::Tensor& tensor) {
  XLATensor xtensor = bridge::GetXlaTensor(tensor);
  return xtensor.GetViewAliasId();
}

std::ptrdiff_t GetTensorId(const at::Tensor& tensor) {
  XLATensor xtensor = bridge::GetXlaTensor(tensor);
  return xtensor.GetUniqueId();
}

std::vector<at::Tensor> GetXlaTensorsFromAten(
    const std::vector<at::Tensor>& aten_tensors,
    const std::vector<std::string>& devices) {
  auto data_handles = CreateTensorsData(aten_tensors, GetXlaDevices(devices));

  std::vector<at::Tensor> xla_tensors;
  xla_tensors.reserve(data_handles.size());
  for (auto& data_handle : data_handles) {
    XLATensor xla_tensor = XLATensor::Create(std::move(data_handle));
    xla_tensors.push_back(bridge::AtenFromXlaTensor(std::move(xla_tensor)));
  }
  return xla_tensors;
}

std::shared_ptr<ir::Value> CreateToken(const std::string& device_str) {
  // This should be using xla::CreateToken() once we have added Token support to
  // XLA AllReduce(). Meanwhile we use a constant as token, and we handle it
  // accordingly in cross_replica_reduces.cpp.
  // This needs to be device data (hence coming in as XLA computation parameter)
  // as otherwise the XLA compiler passes will remove it, vanishing its
  // sequencing effects.
  Device device = GetDeviceOrCurrent(device_str);
  ir::Value ir_value =
      XLATensor::GetDeviceDataIrValue(0.0, xla::PrimitiveType::F32, device);
  return std::make_shared<ir::Value>(std::move(ir_value));
}

at::Tensor GetXlaTensorDimensionSize(const at::Tensor& tensor, xla::int64 dim) {
  XLATensor xtensor = bridge::GetXlaTensor(tensor);
  return bridge::AtenFromXlaTensor(
      XLATensor::get_dimensions_size(xtensor, {dim}));
}

py::object GetMetricData(const std::string& name) {
  xla::metrics::MetricData* data = xla::metrics::GetMetric(name);
  if (data == nullptr) {
    return py::none();
  }

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

py::object GetRevisions() {
  auto py_dict = py::dict();
  py_dict["xla"] = std::string(XLA_GITREV);
  py_dict["torch"] = std::string(TORCH_GITREV);
  return py_dict;
}

std::vector<py::bytes> Rendezvous(int ordinal, const std::string& tag,
                                  const std::string& payload,
                                  const std::vector<xla::int64>& replicas) {
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
    std::string path, const std::string& compression, xla::int64 buffer_size) {
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

std::unique_ptr<tensorflow::RandomAccessFile> OpenTfFile(
    const std::string& path) {
  tensorflow::Env* env = tensorflow::Env::Default();
  std::unique_ptr<tensorflow::RandomAccessFile> file;
  XLA_CHECK_OK(env->NewRandomAccessFile(path, &file));
  return file;
}

py::object StatTfFile(const std::string& path) {
  tensorflow::Env* env = tensorflow::Env::Default();
  tensorflow::FileStatistics stat;
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

py::bytes ReadTfFile(tensorflow::RandomAccessFile* file, uint64_t offset,
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

        tensorflow::StringPiece result;
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

std::unique_ptr<tensorflow::WritableFile> CreateTfFile(
    const std::string& path) {
  tensorflow::Env* env = tensorflow::Env::Default();
  std::unique_ptr<tensorflow::WritableFile> file;
  XLA_CHECK_OK(env->NewWritableFile(path, &file));
  return file;
}

void WriteTfFile(tensorflow::WritableFile* file, const std::string& data) {
  XLA_CHECK_OK(file->Append(tensorflow::StringPiece(data.data(), data.size())));
}

void FlushTfFile(tensorflow::WritableFile* file) {
  XLA_CHECK_OK(file->Flush());
  XLA_CHECK_OK(file->Sync());
}

py::object ListTfFs(const std::string& pattern) {
  std::vector<std::string> files;
  {
    NoGilSection nogil;
    tensorflow::Env* env = tensorflow::Env::Default();
    XLA_CHECK_OK(env->GetMatchingPaths(pattern, &files));
  }

  auto py_files = py::tuple(files.size());
  for (size_t i = 0; i < files.size(); ++i) {
    py_files[i] = files[i];
  }
  return py_files;
}

void RemoveTfFile(const std::string& path) {
  tensorflow::Env* env = tensorflow::Env::Default();
  XLA_CHECK_OK(env->DeleteFile(path));
}

py::object XlaNms(const at::Tensor& boxes, const at::Tensor& scores,
                  const at::Tensor& score_threshold,
                  const at::Tensor& iou_threshold, xla::int64 output_size) {
  at::Tensor selected_indices;
  at::Tensor num_valid;
  {
    NoGilSection nogil;
    auto nms_result = XLATensor::nms(
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
  std::vector<XLATensor> xinputs = GetXlaTensors(inputs, /*want_all=*/true);
  std::vector<XLATensor> xresults =
      XLATensor::user_computation(opname, xinputs, std::move(computation));
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
  Device device = GetDeviceOrCurrent(device_str);
  return CreateComputationShapeFromTensor(tensor, &device);
}

py::dict GetMemoryInfo(const std::string& device_str) {
  xla::ComputationClient::MemoryInfo mem_info;
  {
    NoGilSection nogil;
    Device device = GetDeviceOrCurrent(device_str);
    mem_info = xla::ComputationClient::Get()->GetMemoryInfo(device.ToString());
  }
  auto py_dict = py::dict();
  py_dict["kb_free"] = mem_info.kb_free;
  py_dict["kb_total"] = mem_info.kb_total;
  return py_dict;
}

// Must be called holding GIL as it reads Python objects. Also, Python objects
// are reference counted; reading py::dict will increase its reference count.
absl::flat_hash_map<std::string, absl::variant<int>> ConvertDictToMap(
    const py::dict& dict) {
  absl::flat_hash_map<std::string, absl::variant<int>> map;
  for (const auto& kw : dict) {
    if (!kw.second.is_none()) {
      map.emplace(kw.first.cast<std::string>(), kw.second.cast<int>());
    }
  }
  return map;
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

  profiler.def("trace",
               [](const char* service_addr, const char* logdir, int duration_ms,
                  int num_tracing_attempts, py::dict options) {
                 absl::flat_hash_map<std::string, absl::variant<int>> opts =
                     ConvertDictToMap(options);
                 tensorflow::Status status;
                 {
                   NoGilSection nogil;
                   status = tensorflow::profiler::pywrap::Trace(
                       service_addr, logdir, /*worker_list=*/"",
                       /*include_dataset_ops=*/false, duration_ms,
                       num_tracing_attempts, opts);
                 }
                 if (!status.ok()) {
                   PyErr_SetString(PyExc_RuntimeError, status.error_message());
                   throw py::error_already_set();
                 }
               },
               py::arg("service_addr"), py::arg("logdir"),
               py::arg("duration_ms") = 1000,
               py::arg("num_tracing_attempts") = 3, py::arg("options"));

  py::class_<tensorflow::profiler::TraceMeWrapper> traceme_class(
      profiler, "TraceMe", py::module_local());
  traceme_class.def(py::init<py::str, py::kwargs>())
      .def("__enter__", [](py::object self) -> py::object { return self; })
      .def("__exit__",
           [](py::object self, const py::object& ex_type,
              const py::object& ex_value,
              const py::object& traceback) -> py::object {
             py::cast<tensorflow::profiler::TraceMeWrapper*>(self)->Stop();
             return py::none();
           })
      .def("set_metadata", &tensorflow::profiler::TraceMeWrapper::SetMetadata)
      .def_static("is_enabled",
                  &tensorflow::profiler::TraceMeWrapper::IsEnabled);

  py::class_<ir::ScopePusher, std::unique_ptr<ir::ScopePusher>>
      scope_pusher_class(profiler, "ScopePusher");
  profiler.def("scope_pusher",
               [](const std::string& name) -> std::unique_ptr<ir::ScopePusher> {
                 return absl::make_unique<ir::ScopePusher>(name);
               });
}

void InitXlaModuleBindings(py::module m) {
  m.def("_initialize_aten_bindings",
        []() { AtenXlaType::InitializeAtenBindings(); });
  m.def("_prepare_to_exit", []() { PrepareToExit(); });
  m.def("_get_git_revs", []() { return GetRevisions(); });
  m.def("_get_xla_tensor_dimension_size",
        [](const at::Tensor& tensor, int dim) {
          return GetXlaTensorDimensionSize(tensor, dim);
        });
  m.def("_xla_nms", [](const at::Tensor& boxes, const at::Tensor& scores,
                       const at::Tensor& score_threshold,
                       const at::Tensor& iou_threshold,
                       xla::int64 output_size) {
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
          auto coverter = [](absl::Span<const ir::Node* const> nodes) {
            return ir::DumpUtil::ToDot(nodes);
          };
          return GetTensorsDump(tensors, coverter);
        });
  m.def("_get_xla_tensors_text",
        [](const std::vector<at::Tensor>& tensors) -> std::string {
          auto coverter = [](absl::Span<const ir::Node* const> nodes) {
            return ir::DumpUtil::ToText(nodes);
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
           const std::vector<xla::int64>& replicas) {
          return Rendezvous(ordinal, tag, payload, replicas);
        });

  py::class_<ir::Value, std::shared_ptr<ir::Value>>(m, "IrValue");
  m.def("_xla_create_token",
        [](const std::string& device) { return CreateToken(device); });
  m.def("_xla_all_reduce_inplace", [](const std::string& reduce_type,
                                      const std::vector<at::Tensor>& tensors,
                                      const std::shared_ptr<ir::Value>& token,
                                      double scale, const py::list& groups) {
    std::vector<std::vector<xla::int64>> replica_groups =
        CreateReduceGroups(groups);
    std::shared_ptr<ir::Value> new_token;
    {
      NoGilSection nogil;
      new_token =
          AllReduceInPlace(reduce_type, tensors, token, scale, replica_groups);
    }
    return new_token;
  });
  m.def("_xla_all_reduce",
        [](const std::string& reduce_type, const at::Tensor& input,
           const std::shared_ptr<ir::Value>& token, double scale,
           const py::list& groups) {
          std::vector<std::vector<xla::int64>> replica_groups =
              CreateReduceGroups(groups);
          at::Tensor result;
          std::shared_ptr<ir::Value> new_token;
          {
            NoGilSection nogil;
            std::tie(result, new_token) =
                AllReduce(reduce_type, input, token, scale, replica_groups);
          }
          auto result_tuple = py::tuple(2);
          result_tuple[0] = torch::autograd::make_variable(
              result, /*requires_grad=*/input.requires_grad());
          result_tuple[1] = new_token;
          return result_tuple;
        });
  m.def("_xla_all_to_all",
        [](const at::Tensor& input, const std::shared_ptr<ir::Value>& token,
           xla::int64 split_dimension, xla::int64 concat_dimension,
           xla::int64 split_count, const py::list& groups) {
          std::vector<std::vector<xla::int64>> replica_groups =
              CreateReduceGroups(groups);
          at::Tensor result;
          std::shared_ptr<ir::Value> new_token;
          {
            NoGilSection nogil;
            std::tie(result, new_token) =
                AllToAll(input, token, split_dimension, concat_dimension,
                         split_count, replica_groups);
          }
          auto result_tuple = py::tuple(2);
          result_tuple[0] = torch::autograd::make_variable(
              result, /*requires_grad=*/input.requires_grad());
          result_tuple[1] = new_token;
          return result_tuple;
        });
  m.def("_xla_collective_permute",
        [](const at::Tensor& input, const std::shared_ptr<ir::Value>& token,
           const py::list& pairs) {
          std::vector<std::pair<xla::int64, xla::int64>> source_target_pairs =
              CreateSourceTargetPairs(pairs);
          at::Tensor result;
          std::shared_ptr<ir::Value> new_token;
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
  m.def("_xla_set_default_device", [](const std::string& device) {
    return SetCurrentThreadDevice(device);
  });
  m.def("_xla_get_default_device", []() { return GetCurrentThreadDevice(); });
  m.def("_xla_set_rng_seed",
        [](xla::uint64 seed, const std::string& device) {
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
          XLATensor::WaitDeviceOps(devices);
        },
        py::arg("devices"));
  m.def("_xla_counter_names", []() { return xla::metrics::GetCounterNames(); });
  m.def("_xla_counter_value", [](const std::string& name) -> py::object {
    xla::metrics::CounterData* data = xla::metrics::GetCounter(name);
    return data != nullptr ? py::cast<int64_t>(data->Value()) : py::none();
  });
  m.def("_xla_metric_names", []() { return xla::metrics::GetMetricNames(); });
  m.def("_xla_metric_data", [](const std::string& name) -> py::object {
    return GetMetricData(name);
  });
  m.def("_xla_metrics_report",
        []() { return xla::metrics_reader::CreateMetricReport(); });
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
           xla::int64 buffer_size) {
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

  py::class_<tensorflow::RandomAccessFile>(m, "TfRdFile");
  m.def("_xla_tffile_open", [](const std::string& path) {
    std::unique_ptr<tensorflow::RandomAccessFile> file;
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
        [](tensorflow::RandomAccessFile* file, uint64_t offset, size_t size) {
          return ReadTfFile(file, offset, size);
        });

  py::class_<tensorflow::WritableFile>(m, "TfWrFile");
  m.def("_xla_tffile_create", [](const std::string& path) {
    std::unique_ptr<tensorflow::WritableFile> file;
    {
      NoGilSection nogil;
      file = CreateTfFile(path);
    }
    return py::cast(file.release(),
                    pybind11::return_value_policy::take_ownership);
  });
  m.def("_xla_tffile_write",
        [](tensorflow::WritableFile* file, const std::string& data) {
          NoGilSection nogil;
          WriteTfFile(file, data);
        });
  m.def("_xla_tffile_flush", [](tensorflow::WritableFile* file) {
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
  m.def("_xla_op_param", [](op_builder::BuilderPtr builder, xla::int64 param_no,
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

  BuildProfilerSubmodule(&m);
}

}  // namespace

void InitXlaBindings(py::module m) { InitXlaModuleBindings(m); }

}  // namespace torch_xla

PYBIND11_MODULE(_XLAC, m) { torch_xla::InitXlaBindings(m); }
