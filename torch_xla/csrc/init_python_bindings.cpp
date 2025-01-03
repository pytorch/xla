#include <ATen/dlpack.h>
#include <Python.h>
#include <c10/core/Device.h>
#include <google/protobuf/text_format.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>

#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/variant.h"
#include "pybind11/attr.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/functional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
#include "torch_xla/csrc/XLANativeFunctions.h"
#include "torch_xla/csrc/aten_autograd_ops.h"
#include "torch_xla/csrc/aten_fallback.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/dl_convertor.h"
#include "torch_xla/csrc/dtype.h"
#include "torch_xla/csrc/dynamic_shape_detector.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/layout_manager.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/metrics.h"
#include "torch_xla/csrc/runtime/metrics_analysis.h"
#include "torch_xla/csrc/runtime/metrics_reader.h"
#include "torch_xla/csrc/runtime/pjrt_computation_client.h"
#include "torch_xla/csrc/runtime/pjrt_registry.h"
#include "torch_xla/csrc/runtime/profiler.h"
#include "torch_xla/csrc/runtime/runtime.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/runtime/util.h"
#include "torch_xla/csrc/runtime/xla_coordinator.h"
#include "torch_xla/csrc/runtime/xla_util.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/tensor_impl.h"
#include "torch_xla/csrc/tensor_methods.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/version.h"
#include "torch_xla/csrc/xla_backend_impl.h"
#include "torch_xla/csrc/xla_graph_executor.h"
#include "torch_xla/csrc/xla_op_builder.h"
#include "torch_xla/csrc/xla_sharding_util.h"
#include "tsl/platform/env.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/python/profiler/internal/traceme_wrapper.h"

namespace torch_xla {
namespace {

static int64_t seed_info_id = -127389;

struct NoGilSection {
  NoGilSection() : state(PyEval_SaveThread()) {}
  ~NoGilSection() { PyEval_RestoreThread(state); }
  PyThreadState* state = nullptr;
};

class PyPjRtPlugin : public runtime::PjRtPlugin {
 public:
  using runtime::PjRtPlugin::PjRtPlugin;

  std::string library_path() const override {
    PYBIND11_OVERRIDE_PURE(std::string, runtime::PjRtPlugin, library_path, );
  }

  // Templates with commas confuse pybind's macros, so use an alias here
  // See https://github.com/pybind/pybind11/issues/2185#issuecomment-634005168
  using PjRtCreateOptions = std::unordered_map<std::string, xla::PjRtValueType>;
  const PjRtCreateOptions client_create_options() const override {
    PYBIND11_OVERRIDE_PURE(PjRtCreateOptions, runtime::PjRtPlugin,
                           client_create_options, );
  }

  bool requires_xla_coordinator() const override {
    PYBIND11_OVERRIDE_PURE(bool, runtime::PjRtPlugin,
                           requires_xla_coordinator, );
  }
};

std::optional<torch::lazy::BackendDevice> GetOptionalDevice(
    const std::string& device_str) {
  if (device_str.empty()) {
    return std::nullopt;
  }
  return bridge::AtenDeviceToXlaDevice(c10::Device(device_str));
}

torch::lazy::BackendDevice GetDeviceOrCurrent(const std::string& device_str) {
  if (device_str.empty()) {
    return bridge::GetCurrentDevice();
  }
  return bridge::AtenDeviceToXlaDevice(c10::Device(device_str));
}

void WaitDeviceOps(absl::Span<const std::string> devices = {}) {
  XLAGraphExecutor::Get()->WaitDeviceOps(devices);
  runtime::GetComputationClient()->WaitDeviceOps(devices);
}

void PrepareToExit() {
  runtime::ComputationClient* client =
      runtime::GetComputationClientIfInitialized();
  if (client != nullptr) {
    auto xla_device = GetDeviceOrCurrent("");
    SetAllReduceToken(xla_device, nullptr);
    WaitDeviceOps();
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

bool IsNonDeviceDataIR(const at::Tensor& tensor) {
  XLATensorPtr xtensor = bridge::GetXlaTensor(tensor);
  return xtensor->CurrentIrValue() &&
         !DeviceData::Cast(xtensor->CurrentIrValue().node.get());
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

std::vector<at::Tensor> XlaCustomCall(
    const std::vector<at::Tensor>& inputs, const std::string& payload,
    const std::vector<std::vector<int64_t>>& output_shapes,
    const std::vector<py::object>& output_dtypes, bool is_tpu) {
  std::vector<at::ScalarType> dtypes;
  dtypes.reserve(output_dtypes.size());
  for (auto& dtype : output_dtypes) {
    dtypes.push_back(reinterpret_cast<THPDtype*>(dtype.ptr())->scalar_type);
  }

  if (is_tpu) {
    return bridge::AtenFromXlaTensors(tensor_methods::tpu_custom_call(
        bridge::GetXlaTensors(inputs), payload, output_shapes, dtypes));
  }
  return bridge::AtenFromXlaTensors(tensor_methods::gpu_custom_call(
      bridge::GetXlaTensors(inputs), payload, output_shapes, dtypes));
}

std::vector<std::vector<int>> ExtractXlaDotGeneralDimVectors(
    const py::tuple& dimension_numbers) {
  // Expect Python arg `dimension_numbers` to be
  // (([lhs_contract_dim0,...], [rhs_contract_dim0,...]), ([lhs_batch_dim0,...],
  // [rhs_batch_dim0,...]))
  std::vector<std::vector<int>> dim_vectors;
  XLA_CHECK_EQ(dimension_numbers.size(), 2)
      << "dimension_numbers must be a tuple of 2 elements";
  for (int i = 0; i < 2; ++i) {
    XLA_CHECK(py::isinstance<py::tuple>(dimension_numbers[i]))
        << "_xla_dot_general: Dimension_numbers[" << i << "] must be a tuple";
    py::tuple cast_tuple = py::cast<py::tuple>(dimension_numbers[i]);
    XLA_CHECK(cast_tuple.size() == 0 || cast_tuple.size() == 2)
        << "_xla_dot_general: Contracting/batch dims must be speficied for "
           "both lhs and rhs or neither";
    if (cast_tuple.size() == 0) {
      // Empty tuple means no contracting/batch dims
      dim_vectors.push_back({});
      dim_vectors.push_back({});
    } else {
      for (const auto& dim_list : cast_tuple) {
        XLA_CHECK(py::isinstance<py::list>(dim_list))
            << "_xla_dot_general: lhs/rhs contracting/batch dims must be a "
               "list";
        const py::list& dim_list_casted = py::cast<py::list>(dim_list);
        std::vector<int> dim_vector;
        for (const auto& item : dim_list_casted) {
          XLA_CHECK(py::isinstance<py::int_>(item))
              << "_xla_dot_general: lhs/rhs contracting/batch dims must be a "
                 "list of integers";
          dim_vector.push_back(py::cast<int>(item));
        }
        dim_vectors.push_back(dim_vector);
      }
    }
  }
  return dim_vectors;
}

at::Tensor XlaDotGeneral(const at::Tensor& lhs, const at::Tensor& rhs,
                         const std::vector<std::vector<int>>& dim_vectors,
                         std::optional<py::object> preferred_element_type) {
  std::optional<at::ScalarType> at_preferred_element_type;
  if (preferred_element_type.has_value()) {
    at_preferred_element_type =
        reinterpret_cast<THPDtype*>(preferred_element_type.value().ptr())
            ->scalar_type;
  }
  return bridge::AtenFromXlaTensor(tensor_methods::xla_dot_general(
      bridge::GetXlaTensor(lhs), bridge::GetXlaTensor(rhs), dim_vectors,
      at_preferred_element_type));
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

void AllReduceInPlace(const std::string& reduce_type,
                      const std::vector<at::Tensor>& tensors, double scale,
                      const std::vector<std::vector<int64_t>>& replica_groups,
                      bool pin_layout) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  std::vector<XLATensorPtr> xtensors =
      GetXlaTensors(tensors, /*want_all=*/true);
  tensor_methods::all_reduce(xtensors, GetReduceType(reduce_type), scale,
                             replica_groups, pin_layout);
  std::vector<XLATensorPtr> new_xtensors =
      GetXlaTensors(tensors, /*want_all=*/true);
  bridge::ReplaceXlaTensor(tensors, new_xtensors);
}

at::Tensor AllReduce(const std::string& reduce_type, const at::Tensor& input,
                     double scale,
                     const std::vector<std::vector<int64_t>>& replica_groups,
                     bool pin_layout) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto result = tensor_methods::all_reduce(bridge::GetXlaTensor(input),
                                           GetReduceType(reduce_type), scale,
                                           replica_groups, pin_layout);
  return bridge::AtenFromXlaTensor(std::move(result));
}

at::Tensor DynamicExpand(const at::Tensor& input,
                         const std::vector<int64_t>& size,
                         const at::Tensor& src_tensor, int src_dim,
                         int target_dim) {
  XLATensorPtr result = tensor_methods::dynamic_expand(
      bridge::GetXlaTensor(input), size, bridge::GetXlaTensor(src_tensor),
      src_dim, target_dim);
  return bridge::AtenFromXlaTensor(std::move(result));
}

at::Tensor DynamicView(const at::Tensor& input,
                       const std::vector<int64_t>& size,
                       const at::Tensor& src_tensor, int src_dim,
                       int target_dim, float mul_scaler) {
  XLATensorPtr result = tensor_methods::dynamic_view(
      bridge::GetXlaTensor(input), size, bridge::GetXlaTensor(src_tensor),
      src_dim, target_dim, mul_scaler);
  return bridge::AtenFromXlaTensor(std::move(result));
}

at::Tensor CastInt4(const at::Tensor& weight,
                    const std::vector<int>& int4_weight_values) {
  auto result = tensor_methods::cast_int4(bridge::GetXlaTensor(weight),
                                          int4_weight_values);
  return bridge::AtenFromXlaTensor(std::move(result));
}

at::Tensor QuantizeTensor(const at::Tensor& input,
                          const std::vector<float>& scale_list,
                          const std::vector<int>& zero_point_list,
                          int quant_min, int quant_max,
                          const std::string& dtype, int axis) {
  auto result = tensor_methods::quantize_tensor(
      bridge::GetXlaTensor(input), scale_list, zero_point_list, quant_min,
      quant_max, dtype, axis);
  return bridge::AtenFromXlaTensor(std::move(result));
}

at::Tensor DequantizeTensor(const at::Tensor& input,
                            const std::vector<float>& scale_list,
                            const std::vector<int>& zero_point_list,
                            int quant_min, int quant_max,
                            const std::string& dtype, int axis) {
  auto result = tensor_methods::dequantize_tensor(
      bridge::GetXlaTensor(input), scale_list, zero_point_list, quant_min,
      quant_max, dtype, axis);
  return bridge::AtenFromXlaTensor(std::move(result));
}

std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>> ReduceScatter(
    const std::string& reduce_type, const at::Tensor& input,
    const std::shared_ptr<torch::lazy::Value>& token, double scale,
    int64_t scatter_dim, int64_t shard_count,
    const std::vector<std::vector<int64_t>>& replica_groups, bool pin_layout) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
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
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr out = bridge::GetXlaTensor(output);
  torch::lazy::Value new_token;
  new_token = tensor_methods::reduce_scatter_out(
      out, bridge::GetXlaTensor(input), *token, GetReduceType(reduce_type),
      scale, scatter_dim, shard_count, replica_groups, pin_layout);
  return std::make_shared<torch::lazy::Value>(new_token);
}

std::pair<std::vector<at::Tensor>, std::shared_ptr<torch::lazy::Value>>
ReduceScatterCoalesced(const std::string& reduce_type,
                       const std::vector<at::Tensor>& inputs,
                       const std::shared_ptr<torch::lazy::Value>& token,
                       double scale, int64_t scatter_dim, int64_t shard_count,
                       const std::vector<std::vector<int64_t>>& replica_groups,
                       bool pin_layout) {
  std::vector<XLATensorPtr> xtensors = GetXlaTensors(inputs, /*want_all=*/true);
  std::vector<XLATensorPtr> result;
  torch::lazy::Value new_token;
  std::tie(result, new_token) = tensor_methods::reduce_scatter_coalesced(
      xtensors, *token, GetReduceType(reduce_type), scale, scatter_dim,
      shard_count, replica_groups, pin_layout);
  std::vector<at::Tensor> aten_result;
  for (auto& xt : result) {
    aten_result.emplace_back(bridge::AtenFromXlaTensor(std::move(xt)));
  }
  return {aten_result, std::make_shared<torch::lazy::Value>(new_token)};
}

std::shared_ptr<torch::lazy::Value> ReduceScatterCoalescedOut(
    const std::string& reduce_type, std::vector<at::Tensor>& outputs,
    const std::vector<at::Tensor>& inputs,
    const std::shared_ptr<torch::lazy::Value>& token, double scale,
    int64_t scatter_dim, int64_t shard_count,
    const std::vector<std::vector<int64_t>>& replica_groups, bool pin_layout) {
  std::vector<XLATensorPtr> xtensors_out =
      GetXlaTensors(outputs, /*want_all=*/true);
  std::vector<XLATensorPtr> xtensors = GetXlaTensors(inputs, /*want_all=*/true);
  torch::lazy::Value new_token;
  new_token = tensor_methods::reduce_scatter_coalesced_out(
      xtensors_out, xtensors, *token, GetReduceType(reduce_type), scale,
      scatter_dim, shard_count, replica_groups, pin_layout);
  return std::make_shared<torch::lazy::Value>(new_token);
}

at::Tensor AllGather(const at::Tensor& input, int64_t dim, int64_t shard_count,
                     const std::vector<std::vector<int64_t>>& replica_groups,
                     bool pin_layout) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto result =
      tensor_methods::all_gather(bridge::GetXlaTensor(input), dim, shard_count,
                                 replica_groups, pin_layout);
  return bridge::AtenFromXlaTensor(std::move(result));
}

std::shared_ptr<torch::lazy::Value> AllGatherOut(
    at::Tensor& output, const at::Tensor& input,
    const std::shared_ptr<torch::lazy::Value>& token, int64_t dim,
    int64_t shard_count,
    const std::vector<std::vector<int64_t>>& replica_groups, bool pin_layout) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr out = bridge::GetXlaTensor(output);
  torch::lazy::Value new_token;
  new_token = tensor_methods::all_gather_out(out, bridge::GetXlaTensor(input),
                                             *token, dim, shard_count,
                                             replica_groups, pin_layout);
  return std::make_shared<torch::lazy::Value>(new_token);
}

std::pair<std::vector<at::Tensor>, std::shared_ptr<torch::lazy::Value>>
AllGatherCoalesced(const std::vector<at::Tensor>& tensors,
                   const std::shared_ptr<torch::lazy::Value>& token,
                   int64_t dim, int64_t shard_count,
                   const std::vector<std::vector<int64_t>>& replica_groups,
                   bool pin_layout) {
  std::vector<XLATensorPtr> xtensors =
      GetXlaTensors(tensors, /*want_all=*/true);
  std::vector<XLATensorPtr> result;
  torch::lazy::Value new_token;
  std::tie(result, new_token) = tensor_methods::all_gather_coalesced(
      xtensors, *token, dim, shard_count, replica_groups, pin_layout);
  std::vector<at::Tensor> aten_result;
  for (auto& xt : result) {
    aten_result.emplace_back(bridge::AtenFromXlaTensor(std::move(xt)));
  }
  return {aten_result, std::make_shared<torch::lazy::Value>(new_token)};
}

std::shared_ptr<torch::lazy::Value> AllGatherCoalescedOut(
    std::vector<at::Tensor>& outputs, const std::vector<at::Tensor>& inputs,
    const std::shared_ptr<torch::lazy::Value>& token, int64_t dim,
    int64_t shard_count,
    const std::vector<std::vector<int64_t>>& replica_groups, bool pin_layout) {
  std::vector<XLATensorPtr> xtensors_out =
      GetXlaTensors(outputs, /*want_all=*/true);
  std::vector<XLATensorPtr> xtensors = GetXlaTensors(inputs, /*want_all=*/true);
  torch::lazy::Value new_token;
  new_token = tensor_methods::all_gather_coalesced_out(
      xtensors_out, xtensors, *token, dim, shard_count, replica_groups,
      pin_layout);
  return std::make_shared<torch::lazy::Value>(new_token);
}

std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>> AllToAll(
    const at::Tensor& input, const std::shared_ptr<torch::lazy::Value>& token,
    int64_t split_dimension, int64_t concat_dimension, int64_t split_count,
    const std::vector<std::vector<int64_t>>& replica_groups, bool pin_layout) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
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
                const std::vector<std::string>& devices, bool wait,
                bool reset_scope) {
  tsl::profiler::TraceMe activity("StepMarker",
                                  tsl::profiler::TraceMeLevel::kInfo);
  torch::lazy::BackendDevice device = GetDeviceOrCurrent(device_str);
  XLAGraphExecutor::Get()->SyncLiveTensorsGraph(&device, devices, wait);
  XLAGraphExecutor::Get()->MarkStep(device, reset_scope);
  bool debug_mode = runtime::sys_util::GetEnvBool("PT_XLA_DEBUG", false);
  if (TF_PREDICT_FALSE(debug_mode)) {
    std::string report = runtime::metrics::CreatePerformanceReport(
        runtime::GetComputationClient()->GetMetrics());
    if (!report.empty()) {
      std::string fout =
          runtime::sys_util::GetEnvString("PT_XLA_DEBUG_FILE", "");
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

std::string GetTensorsHloGraph(const std::vector<at::Tensor>& tensors,
                               EmitMode mode) {
  std::vector<XLATensorPtr> xtensors =
      GetXlaTensors(tensors, /*want_all=*/false);
  return XLAGraphExecutor::Get()->DumpHloComputation(xtensors, mode);
}

std::string GetXLAShardingSpec(const XLATensorPtr xtensor) {
  auto sharding_spec = xtensor->sharding_spec();
  if (sharding_spec != nullptr) {
    auto hlo_sharding = xla::HloSharding::FromProto(sharding_spec->sharding);
    return hlo_sharding->ToString();
  }
  return std::string();
}

std::string GetXLATensorDebugInfo(const at::Tensor& tensor) {
  auto xtensor = bridge::TryGetXlaTensor(tensor);
  if (!xtensor) {
    return "Not a XLATensor\n";
  }
  std::stringstream ss;
  ss << "XLATensor {\n";
  ss << "TensorID: " << xtensor->GetUniqueId() << "\n";
  ss << "Device: " << xtensor->GetDevice() << "\n";
  ss << "XLA Shape: " << xtensor->shape().get().ToString() << "\n";

  std::string sharding_spec_str = GetXLAShardingSpec(xtensor);
  ss << "ShardingSpec: "
     << ((sharding_spec_str.size() > 0) ? sharding_spec_str : "None");
  ss << "\n";

  torch::lazy::Value ir_value = xtensor->CurrentIrValue();
  ss << "IR: ";
  if (ir_value) {
    ss << ir_value.node->ToString() << "\n";
  } else {
    ss << "None\n";
  }

  torch::lazy::BackendDataPtr handle = xtensor->CurrentDataHandle();
  if (handle) {
    auto data =
        std::dynamic_pointer_cast<runtime::ComputationClient::Data>(handle);
    ss << data->ToString();
  } else {
    ss << "XLAData: None\n";
  }

  auto at_tensor = xtensor->CurrentTensorData();
  ss << "Tensor on host: ";
  if (at_tensor) {
    ss << "with size " << at_tensor->sizes() << "\n";
  } else {
    ss << "None\n";
  }

  ss << "}\n";
  return ss.str();
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
    const std::vector<std::string>& devices,
    const std::optional<std::vector<XLATensor::ShardingSpecPtr>>
        sharding_specs) {
  std::vector<std::shared_ptr<torch::lazy::BackendData>> data_handles;
  if (sharding_specs.has_value()) {
    data_handles = CreateTensorsData(aten_tensors, sharding_specs.value(),
                                     GetXlaDevices(devices));
  } else {
    data_handles = CreateTensorsData(aten_tensors, GetXlaDevices(devices));
  }

  std::vector<at::Tensor> xla_tensors;
  xla_tensors.reserve(data_handles.size());
  for (int i = 0; i < data_handles.size(); i++) {
    auto& data_handle = data_handles[i];
    XLATensorPtr xla_tensor = XLATensor::Create(std::move(data_handle));
    if (sharding_specs.has_value() && sharding_specs.value()[i] != nullptr) {
      xla_tensor->SetShardingSpec(*sharding_specs.value()[i]);
    }
    xla_tensors.push_back(bridge::AtenFromXlaTensor(std::move(xla_tensor)));
  }
  return xla_tensors;
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
  if (auto* data = runtime::metrics::GetMetric(name)) {
    return GetMetricData<runtime::metrics::MetricData>(data);
  }
  return py::none();
}

py::object GetRevisions() {
  auto py_dict = py::dict();
  py_dict["xla"] = std::string(XLA_GITREV);
  py_dict["torch"] = std::string(TORCH_GITREV);
  return py_dict;
}

std::vector<at::Tensor> XlaUserComputation(
    const std::string& opname, const std::vector<at::Tensor>& inputs,
    runtime::ComputationClient::ComputationPtr computation) {
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

runtime::ComputationClient::ComputationPtr CreateComputation(
    const std::string& name, xla::XlaOp root) {
  xla::XlaComputation computation = ConsumeValue(root.builder()->Build(root));
  return std::make_shared<runtime::ComputationClient::Computation>(
      name, std::move(computation));
}

runtime::ComputationClient::ComputationPtr CreateComputationFromProto(
    const std::string& name, const std::string& module_proto) {
  xla::HloModuleProto proto;
  proto.ParseFromString(module_proto);
  xla::XlaComputation computation(std::move(proto));
  return std::make_shared<runtime::ComputationClient::Computation>(
      name, std::move(computation));
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
  runtime::ComputationClient::MemoryInfo mem_info;
  {
    NoGilSection nogil;
    torch::lazy::BackendDevice device = GetDeviceOrCurrent(device_str);
    mem_info =
        runtime::GetComputationClient()->GetMemoryInfo(device.toString());
  }
  auto py_dict = py::dict();
  py_dict["bytes_used"] = mem_info.bytes_used;
  py_dict["bytes_limit"] = mem_info.bytes_limit;
  py_dict["peak_bytes_used"] = mem_info.peak_bytes_used;
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
  static bool wants_frames =
      runtime::sys_util::GetEnvBool("XLA_IR_DEBUG", false) |
      runtime::sys_util::GetEnvBool("XLA_HLO_DEBUG", false);
  FLAGS_torch_lazy_ir_debug = wants_frames;
  static bool no_scalars =
      runtime::sys_util::GetEnvBool("XLA_NO_SPECIAL_SCALARS", false);
  FLAGS_torch_lazy_handle_special_scalars = !no_scalars;
  FLAGS_torch_lazy_metrics_samples =
      runtime::sys_util::GetEnvInt("XLA_METRICS_SAMPLES", 1024);
  FLAGS_torch_lazy_metrics_percentiles = runtime::sys_util::GetEnvString(
      "XLA_METRICS_PERCENTILES", "0.01:0.05:0.1:0.2:0.5:0.8:0.9:0.95:0.99");
}

at::Tensor MarkTensor(const at::Tensor& input, const std::string& info) {
  XLATensorPtr result =
      tensor_methods::mark_tensor(bridge::GetXlaTensor(input), info);
  return bridge::AtenFromXlaTensor(std::move(result));
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
    } else if (xtensor->CurrentTensorData().has_value()) {
      need_materialization.push_back(false);
    } else {
      XLA_CHECK(false)
          << "_check_tensor_need_materialization "
             "currently does not handle XLATensor without XLAData and IR";
    }
  }
  return need_materialization;
}

void BuildProfilerSubmodule(py::module* m) {
  py::module profiler = m->def_submodule("profiler", "Profiler integration");
  py::class_<runtime::profiler::ProfilerServer,
             std::unique_ptr<runtime::profiler::ProfilerServer>>
      profiler_server_class(profiler, "ProfilerServer");
  profiler.def(
      "start_server",
      [](int port) -> std::unique_ptr<runtime::profiler::ProfilerServer> {
        auto server = absl::make_unique<runtime::profiler::ProfilerServer>();
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
        absl::Status status;
        {
          NoGilSection nogil;
          for (int i = 0; i <= timeout_s / interval_s; i++) {
            status = runtime::profiler::Trace(service_addr, logdir, duration_ms,
                                              num_tracing_attempts, opts);
            if (status.ok()) {
              return;
            }
            std::this_thread::sleep_for(sleep_s);
          }
        }
        if (!status.ok()) {
          PyErr_SetString(PyExc_RuntimeError, std::string(status.message()));
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
      .def_static("is_enabled", &tsl::profiler::TraceMe::Active);

  py::class_<torch::lazy::ScopePusher,
             std::unique_ptr<torch::lazy::ScopePusher>>
      scope_pusher_class(profiler, "ScopePusher");
  profiler.def(
      "scope_pusher",
      [](const std::string& name) -> std::unique_ptr<torch::lazy::ScopePusher> {
        return absl::make_unique<torch::lazy::ScopePusher>(name);
      });
}

class PyLoweringContext {
 public:
  PyLoweringContext()
      : PyLoweringContext("PyLoweringContext", bridge::GetCurrentDevice()) {}

  PyLoweringContext(const std::string& name)
      : PyLoweringContext(name, bridge::GetCurrentDevice()) {}

  PyLoweringContext(const std::string& name, torch::lazy::BackendDevice device)
      : lowering_ctx(name, device) {}

  // Builds a HLO graph given a set of output tensors.
  void Build(std::vector<at::Tensor> tensors) {
    // Get the backing XLA tensors from the output torch tensor handles
    std::vector<XLATensorPtr> xtensors =
        GetXlaTensors(tensors, /*want_all=*/true);

    // Get the lazy IR value from the output XLA tensors
    std::vector<torch::lazy::Value> ir_values;
    for (auto& xtensor : xtensors) {
      torch::lazy::Value value = xtensor->GetIrValue();
      ir_values.push_back(value);
    }

    // Lower the graph using the output IR values
    for (auto& ir_value : ir_values) {
      xla::XlaOp root = lowering_ctx.GetOutputOp(
          torch::lazy::Output(ir_value.node.get(), ir_value.index));
      lowering_ctx.AddResult(root);
    }

    ShardingUtil::SetHloSharding(&lowering_ctx);

    computation = ConsumeValue(lowering_ctx.BuildXla());
  }

  // Builds a HLO graph given a set of output tensors, and add unused parameters
  // needed in xlacomputation for fori_loop/while_loop.
  void BuildForiLoop(std::vector<at::Tensor> tensors,
                     std::vector<at::Tensor> additional_inputs_list = {}) {
    // Get the backing XLA tensors from the output torch tensor handles
    std::vector<XLATensorPtr> xtensors =
        GetXlaTensors(tensors, /*want_all=*/true);

    // Get the lazy IR value from the output XLA tensors
    std::vector<torch::lazy::Value> ir_values;
    for (auto& xtensor : xtensors) {
      torch::lazy::Value value = xtensor->GetIrValue();
      ir_values.push_back(value);
    }

    // Lower the graph using the output IR values
    for (auto& ir_value : ir_values) {
      xla::XlaOp root = lowering_ctx.GetOutputOp(
          torch::lazy::Output(ir_value.node.get(), ir_value.index));
      lowering_ctx.AddResult(root);
    }

    // add dummy parameter to cond/body xlacomputation's input for xla::while
    // requriement
    if ((GetNameString() == "condctx") or
        (GetNameString() == "bodyctx" && additional_inputs_list.size() != 0)) {
      xla::XlaBuilder* local_builder = lowering_ctx.builder();
      int64_t parameter_idx =
          local_builder->GetProgramShape()->parameters_size();
      int64_t additional_inputs_list_size = additional_inputs_list.size();
      for (int64_t i = parameter_idx; i < additional_inputs_list_size; i++) {
        XLATensorPtr xtensor = bridge::GetXlaTensor(additional_inputs_list[i]);
        xla::Shape shape = xtensor->shape().get();
        xla::XlaOp x = xla::Parameter(local_builder, parameter_idx, shape,
                                      "UnusedArgumentsPlaceholder");
        parameter_idx += 1;
      }
    }

    ShardingUtil::SetHloSharding(&lowering_ctx);

    computation = ConsumeValue(lowering_ctx.BuildXla());

    // wrap inputs of cond/body_computation
    if ((GetNameString() == "condctx") || (GetNameString() == "bodyctx")) {
      std::vector<std::pair<int64_t, int64_t>> input_output_alias_pair;
      std::vector<xla::HloSharding> param_shardings;
      // If sharded, then extract all input Op shardings.
      if (UseVirtualDevice()) {
        param_shardings = XlaHelpers::ExtractInputShardings(computation);
      }
      xla::ProgramShape program_shape =
          ConsumeValue(computation.GetProgramShape());
      // TODO(@manfei): please confirm whether we check for more than two or use
      // default value true
      bool should_wrap_parameter = (program_shape.parameters_size() >= 2);
      if (should_wrap_parameter) {
        computation = ConsumeValue(XlaHelpers::WrapXlaComputation(
            computation, program_shape.parameters(), param_shardings,
            /* buffer_donor_indices */ {}));
      }
    }
  }

  // Get a mapping from the HLO input parameters to the backing Tensor values.
  // This allows the caller to get all parameter information regardless of
  // how the parameter was allocated (inline tensor, nn.Parameter, constant,
  // etc.). This will copy the tensor data from the device to the host.
  std::unordered_map<int64_t, at::Tensor> GetParameterIdTensorMapping() {
    // Find parameters in the lowering
    const std::vector<torch::lazy::BackendDataPtr>& device_data =
        lowering_ctx.GetParametersData();

    // Fetch this parameter data
    std::vector<xla::Literal> literals =
        runtime::GetComputationClient()->TransferFromDevice(
            UnwrapXlaData(device_data));

    // Create a mapping from paramater id to the tensor data
    std::unordered_map<int64_t, at::Tensor> results;
    for (int i = 0; i < device_data.size(); ++i) {
      xla::Literal& literal = literals[i];
      xla::XlaOp op = lowering_ctx.GetParameter(device_data[i]);
      at::ScalarType dtype =
          MaybeUpcastToHostTorchType(literal.shape().element_type());
      at::Tensor input = MakeTensorFromXlaLiteral(literal, dtype);
      std::optional param_id = lowering_ctx.GetParameterId(device_data[i]);
      XLA_CHECK(param_id.has_value());
      results[param_id.value()] = input;
    }
    return results;
  }

  // Returns a mapping from HLO parameter IDs to their corresponding
  // device-backed Tensors. This version only returns parameters that were
  // explicitly allocated on device data, accessible via GetTensorParameterId().
  // Unlike GetParameterIdTensorMapping(), it avoids transferring data from
  // device to host, making it more efficient especially for SPMD scenarios
  // where transferring data involves costly collectives.
  std::unordered_map<int64_t, at::Tensor> GetDeviceParameterIdTensorMapping() {
    // Find parameters in the lowering
    const std::vector<torch::lazy::BackendDataPtr>& device_data =
        lowering_ctx.GetParametersData();

    // Create a mapping from parameter id to the tensor data
    std::unordered_map<int64_t, at::Tensor> param_to_tensor;
    param_to_tensor.reserve(device_data.size());

    for (const auto& data : device_data) {
      std::optional<int64_t> param_id = lowering_ctx.GetParameterId(data);
      XLA_CHECK(param_id.has_value())
          << "Parameter ID must exist for device data";

      at::Tensor tensor =
          bridge::AtenFromXlaTensor(torch_xla::XLATensor::Create(data));
      param_to_tensor.emplace(param_id.value(), std::move(tensor));
    }
    return param_to_tensor;
  }

  // Get the parameter identifier of a given tensor. If the tensor is not a
  // parameter this will always return -1. This is useful in conjunction with
  // GetParameterIdTensorMapping or GetDeviceParameterIdTensorMapping, to
  // identify which values can be baked into the graph and which values must
  // remain parameters.
  int64_t GetTensorParameterId(at::Tensor tensor) {
    // Convert tensor into the backing lazy node
    XLATensorPtr xtensor = bridge::GetXlaTensor(tensor);
    torch::lazy::Value value = xtensor->GetIrValue();
    const torch::lazy::Node* node = value.node.get();
    if (node->op() != xla_device_data) {
      return -1;
    }

    // Convert lazy node data into opaque handle id
    torch::lazy::BackendDataPtr data = DeviceData::Cast(node)->data();
    torch::lazy::BackendData::Handle handle = data->GetHandle();

    // Linearly search parameters and compare opaque handles
    const std::vector<torch::lazy::BackendDataPtr>& device_data =
        lowering_ctx.GetParametersData();
    for (int i = 0; i < device_data.size(); ++i) {
      if (device_data[i]->GetHandle() == handle) {
        std::optional param_id = lowering_ctx.GetParameterId(device_data[i]);
        XLA_CHECK(param_id.has_value());
        return param_id.value();
      }
    }
    return -1;
  }

  // Create a serialized HloModule protobuf from a lowered graph
  py::bytes GetHlo() {
    const xla::HloModuleProto& proto = computation.proto();
    std::string result;
    proto.SerializeToString(&result);
    return result;
  }

  // Create human-readable HloModule protobuf text from a lowered graph
  std::string GetHloText() {
    const xla::HloModuleProto& proto = computation.proto();
    std::string result;
    google::protobuf::TextFormat::PrintToString(proto, &result);
    return result;
  }

  std::string GetHloJsonText() {
    const xla::HloModuleProto& proto = computation.proto();
    std::string result;
    google::protobuf::util::MessageToJsonString(proto, &result);
    return result;
  }

  void SetNameString(const std::string& name) {
    lowering_ctx.set_name_string(name);
  }

  std::string GetNameString() { return lowering_ctx.get_name_string(); }

 private:
  LoweringContext lowering_ctx;
  xla::XlaComputation computation;
};

// Add a submodule which exposes the LoweringContext to python.
void BuildLoweringContextSubmodule(py::module* m) {
  /**
   * Example Python Usage:
   *
   *     import torch
   *     import torch_xla
   *     import torch_xla.core.xla_model as xm
   *
   *     device = xm.xla_device()
   *     example = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
   *
   *     def network(x):
   *         return x + 2.0
   *
   *     result = network(example)
   *
   *     ctx = torch_xla._XLAC.lowering.LoweringContext()
   *     ctx.build([result])
   *     hlo = ctx.hlo()
   *     hlo_text = ctx.hlo_text()
   *     mapping = ctx.parameter_id_tensor_mapping()
   *     input_parameter_id = ctx.tensor_parameter_id(example)
   *
   **/

  py::module lowering =
      m->def_submodule("lowering", "Lowering context and utilities");

  py::class_<PyLoweringContext, std::unique_ptr<PyLoweringContext>>
      lowering_context_class(lowering, "LoweringContext", py::module_local());

  lowering_context_class.def(py::init())
      .def(py::init<std::string>())
      .def("build", &PyLoweringContext::Build)
      .def("buildforiloop", &PyLoweringContext::BuildForiLoop)
      .def("hlo", &PyLoweringContext::GetHlo)
      .def("hlo_text", &PyLoweringContext::GetHloText)
      .def("hlo_json", &PyLoweringContext::GetHloJsonText)
      .def("parameter_id_tensor_mapping",
           &PyLoweringContext::GetParameterIdTensorMapping)
      .def("device_parameter_id_tensor_mapping",
           &PyLoweringContext::GetDeviceParameterIdTensorMapping)
      .def("tensor_parameter_id", &PyLoweringContext::GetTensorParameterId)
      .def("set_name_string", &PyLoweringContext::SetNameString)
      .def("get_name_string", &PyLoweringContext::GetNameString);
}

// Used in the to_dlpack.
void dlPack_Capsule_Destructor(PyObject* data) {
  if (!PyCapsule_IsValid(data, "dltensor")) {
    return;
  }
  DLManagedTensor* dlMTensor =
      static_cast<DLManagedTensor*>(PyCapsule_GetPointer(data, "dltensor"));
  if (dlMTensor) {
    dlMTensor->deleter(dlMTensor);
  } else {
    // The tensor has been deleted. Clear any error from
    // PyCapsule_GetPointer.
    PyErr_Clear();
  }
}

at::Tensor tensor_fromDLPack(PyObject* data) {
  DLManagedTensor* dlMTensor =
      (DLManagedTensor*)PyCapsule_GetPointer(data, "dltensor");
  XLA_CHECK(dlMTensor != nullptr)
      << "from_dlpack received an invalid capsule. Note that a DLTensor "
         "capsule can be consumed only once. You may have already constructed "
         "a tensor from it once.";

  at::Tensor tensor = torch_xla::fromDLPack(dlMTensor);
  PyCapsule_SetName(data, "used_dltensor");
  PyCapsule_SetDestructor(data, nullptr);
  return tensor;
}

void InitXlaModuleBindings(py::module m) {
  m.def("_prepare_to_exit", []() { PrepareToExit(); });
  m.def("_xla_runtime_is_initialized", []() {
    return runtime::GetComputationClientIfInitialized() != nullptr;
  });
  m.def("_xla_computation_cache_is_initialized", []() {
    return XLAGraphExecutor::Get()->IsComputationCacheInitialized();
  });
  m.def("_get_git_revs", []() { return GetRevisions(); });
  m.def("_get_xla_tensor_dimension_size",
        [](const at::Tensor& tensor, int dim) {
          return GetXlaTensorDimensionSize(tensor, dim);
        });
  m.def("_xla_user_computation",
        [](const std::string& opname, const std::vector<at::Tensor>& inputs,
           const runtime::ComputationClient::ComputationPtr& computation) {
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
          return GetTensorsHloGraph(tensors, EmitMode::kHloReadable);
        });
  m.def("_get_xla_tensor_debug_info",
        [](const at::Tensor& tensor) -> std::string {
          return GetXLATensorDebugInfo(tensor);
        });
  m.def("_get_xla_tensor_shape_type",
        [](const at::Tensor& tensor) -> std::string {
          XLATensorPtr xla_tensor = bridge::TryGetXlaTensor(tensor);
          if (xla_tensor) {
            xla::Shape shape = xla_tensor->shape().get();
            return xla::primitive_util::LowercasePrimitiveTypeName(
                shape.element_type());
          }
        });

  py::class_<XLATensor::ShardingSpec, XLATensor::ShardingSpecPtr>(
      m, "XlaShardingSpec")
      .def(py::init([](at::Tensor tensor, const py::list& tile_assignment,
                       const py::list& group_assignment,
                       const py::list& replication_groups, int sharding_type,
                       bool minibatch) {
        xla::Shape global_shape =
            CreateComputationShapeFromTensor(tensor, nullptr);
        if (minibatch) {
          int num_local_devices =
              runtime::GetComputationClient()->GetLocalDevices().size();
          int num_global_devices =
              runtime::GetComputationClient()->GetAllDevices().size();
          XLA_CHECK(tile_assignment.size() == num_global_devices)
              << "Minibatch sharding only supports sharding along the batch "
                 "dimension";
          int batch_dim_shape =
              tensor.sizes()[0] * num_global_devices / num_local_devices;
          global_shape.set_dimensions(0, batch_dim_shape);
        }
        return std::make_shared<XLATensor::ShardingSpec>(
            ShardingUtil::CreateOpSharding(
                tile_assignment, group_assignment, replication_groups,
                ShardingUtil::ShardingType(sharding_type)),
            global_shape, minibatch);
      }));
  m.def(
      "_xla_tensors_from_aten",
      [](const std::vector<at::Tensor>& tensors,
         const std::vector<std::string>& devices,
         const std::optional<std::vector<XLATensor::ShardingSpecPtr>>&
             shardings) {
        std::vector<at::Tensor> result;
        {
          NoGilSection nogil;
          std::vector<at::Tensor> xla_tensors =
              GetXlaTensorsFromAten(tensors, devices, shardings);
          result.reserve(xla_tensors.size());
          for (size_t i = 0; i < xla_tensors.size(); ++i) {
            result.push_back(torch::autograd::make_variable(
                xla_tensors[i],
                /*requires_grad=*/tensors.at(i).requires_grad()));
          }
        }
        return result;
      },
      py::arg("tensors"), py::arg("devices"),
      py::arg("shardings") = py::none());
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
  m.def("_xla_set_auto_sharding", []() {
    ShardingUtil::SetAutoSharding();
    XLA_CHECK(ShardingUtil::GetAutoSharding());
  });
  m.def("_xla_get_auto_sharding",
        []() { return ShardingUtil::GetAutoSharding(); });
  m.def("_xla_get_spmd_config_is_locked", []() { return GetLockSpmdConfig(); });
  m.def("_xla_force_spmd_device", []() {
    // It is actually more easier to force SPMD mode than blocking if there is
    // non-SPMD initialized tensors, for the 3rd-party solution integration. For
    // instance, HuggingFace trainer pre-loads embeddings table and the training
    // initialization is done over multiple scripts. Being able to force SPMD
    // allows the users to call `xr.use_spmd()` more freely, given that the
    // earlier they call, the smaller the one-time overhead of replicating
    // non-SPMD backed tensors.
    torch::lazy::BackendDevice current_device = bridge::GetCurrentDevice();
    std::vector<XLATensorPtr> xtensors =
        XLAGraphExecutor::Get()->GetLiveTensors(&current_device);
    torch::lazy::BackendDevice spmd_device = ParseDeviceString("SPMD:0");
    for (auto xtensor : xtensors) {
      XlaDeviceType xla_device_type =
          static_cast<XlaDeviceType>(xtensor->GetDevice().type());
      if (xla_device_type != XlaDeviceType::SPMD) {
        // Internally this moves the device data to the host and then copy
        // to the SPMD virtual device. The original data should be destroyed
        // in the transition, after creating a detached host-side copy.
        // TODO(yeounoh) Consider CopyToDevice, and make data's device mutable.
        at::Tensor tensor = xtensor->ToTensor(false);
        xtensor->SetXlaData(TensorToXlaData(tensor, spmd_device));
      }
    }

    // Ensure that virtual device is registered.
    XLA_CHECK(UseVirtualDevice(/*force_spmd=*/true));
  });
  m.def("_init_computation_client", []() { runtime::GetComputationClient(); });
  m.def("_xla_get_device_hw_type", [](const at::Tensor& tensor) {
    XLATensorPtr xtensor = bridge::GetXlaTensor(tensor);
    XlaDeviceType xla_device_type =
        static_cast<XlaDeviceType>(xtensor->GetDevice().type());
    return DeviceType(xla_device_type).toString();
  });
  m.def("_xla_get_devices", []() {
    if (UseVirtualDevice()) {
      // Under SPMD context, there is only one virtual devices from user
      // perspective.
      std::vector<std::string> all_devices =
          runtime::GetComputationClient()->GetAllDevices();
      all_devices.resize(1);
      return all_devices;
    } else {
      return runtime::GetComputationClient()->GetLocalDevices();
    }
  });
  m.def("_get_stream_for_cuda_device", [](const int device_id) {
    return runtime::GetComputationClient()->GetCudaStreamForDevice(device_id);
  });
  m.def("_xla_num_devices", []() -> int64_t {
    if (UseVirtualDevice()) {
      return 1;
    } else {
      return runtime::GetComputationClient()->GetNumDevices();
    }
  });
  m.def("_xla_get_all_devices", []() {
    std::vector<std::string> all_devices =
        runtime::GetComputationClient()->GetAllDevices();
    if (UseVirtualDevice()) {
      // Under SPMD context, there is only one virtual devices from user
      // perspective.
      std::vector<std::string> devices = {all_devices[0]};
      return devices;
    } else {
      return all_devices;
    }
  });
  m.def("_xla_get_runtime_devices",
        []() { return runtime::GetComputationClient()->GetLocalDevices(); });
  m.def("_xla_num_runtime_devices", []() -> int64_t {
    return runtime::GetComputationClient()->GetNumDevices();
  });
  m.def("_xla_get_all_runtime_devices", []() {
    std::vector<std::string> all_devices =
        runtime::GetComputationClient()->GetAllDevices();
    return all_devices;
  });
  m.def(
      "_xla_real_devices",
      [](const std::optional<std::vector<std::string>> devices) {
        if (!devices) {
          return runtime::GetComputationClient()->GetLocalDevices();
        }

        std::vector<std::string> xla_devices;
        {
          NoGilSection nogil;
          xla_devices = GetXlaDevices(*devices);
        }
        return xla_devices;
      },
      py::arg("devices") = std::nullopt);
  m.def(
      "_xla_device_kind",
      [](const std::string& device) {
        auto xla_device = bridge::AtenDeviceToXlaDevice(device).toString();
        return runtime::GetComputationClient()->GetDeviceKind(xla_device);
      },
      py::arg("device") = "");
  m.def("_xla_set_replication_devices",
        [](const std::vector<std::string>& devices) {
          auto replication_devices =
              std::make_shared<std::vector<std::string>>(devices);
          runtime::GetComputationClient()->SetReplicationDevices(
              std::move(replication_devices));
        });
  m.def("_xla_get_replication_devices", []() {
    auto replication_devices =
        runtime::GetComputationClient()->GetReplicationDevices();
    return replication_devices != nullptr ? *replication_devices
                                          : std::vector<std::string>();
  });
  m.def("_xla_get_replication_devices_count", []() {
    auto replication_devices =
        runtime::GetComputationClient()->GetReplicationDevices();
    return replication_devices != nullptr ? replication_devices->size() : 0;
  });

  py::class_<torch::lazy::Value, std::shared_ptr<torch::lazy::Value>>(
      m, "IrValue");
  m.def("_xla_all_reduce_inplace", [](const std::string& reduce_type,
                                      const std::vector<at::Tensor>& tensors,
                                      double scale, const py::list& groups,
                                      bool pin_layout) {
    std::vector<std::vector<int64_t>> replica_groups =
        CreateReduceGroups(groups);
    {
      NoGilSection nogil;
      AllReduceInPlace(reduce_type, tensors, scale, replica_groups, pin_layout);
    }
  });
  m.def("_xla_all_reduce", [](const std::string& reduce_type,
                              const at::Tensor& input, double scale,
                              const py::list& groups, bool pin_layout) {
    std::vector<std::vector<int64_t>> replica_groups =
        CreateReduceGroups(groups);
    at::Tensor result;
    {
      NoGilSection nogil;
      result = AllReduce(reduce_type, input, scale, replica_groups, pin_layout);
    }
    return torch::autograd::make_variable(
        result, /*requires_grad=*/input.requires_grad());
  });
  m.def("_xla_spmd_all_reduce", [](const std::string& reduce_type,
                                   const at::Tensor& input, double scale,
                                   const py::list& groups) {
    std::vector<std::vector<int64_t>> replica_groups =
        CreateReduceGroups(groups);
    auto result = tensor_methods::all_reduce(bridge::GetXlaTensor(input),
                                             GetReduceType(reduce_type), scale,
                                             std::move(replica_groups));
    return bridge::AtenFromXlaTensor(std::move(result));
  });
  m.def(
      "_xla_dot_general",
      [](const at::Tensor& lhs, const at::Tensor& rhs,
         py::tuple dimension_numbers,
         std::optional<std::string>& precision_config,
         std::optional<py::object>& preferred_element_type) -> at::Tensor {
        // Python binding for xla::DotGeneral
        // https://openxla.org/xla/operation_semantics#dotgeneral
        std::vector<std::vector<int>> dim_vectors =
            ExtractXlaDotGeneralDimVectors(dimension_numbers);
        XLA_CHECK(!precision_config.has_value())
            << "_xla_dot_general: precision_config is not supported yet, "
               "default precision setting will be applied.";
        at::Tensor result =
            XlaDotGeneral(lhs, rhs, dim_vectors, preferred_element_type);
        return result;
      },
      py::arg("lhs"), py::arg("rhs"), py::arg("dimension_numbers"),
      py::arg("precision_config") = py::none(),
      py::arg("preferred_element_type") = py::none());
  m.def("_xla_cast_int4",
        [](const at::Tensor& weight,
           const std::vector<int>& int4_weight_values) -> at::Tensor {
          at::Tensor result;
          {
            NoGilSection nogil;
            result = CastInt4(weight, int4_weight_values);
          }
          return result;
        });
  m.def("_xla_quantize_tensor",
        [](const at::Tensor& input, const std::vector<float>& scale_list,
           const std::vector<int>& zero_point_list, int quant_min,
           int quant_max, const std::string& dtype, int axis) -> at::Tensor {
          at::Tensor result;
          {
            NoGilSection nogil;
            result = QuantizeTensor(input, scale_list, zero_point_list,
                                    quant_min, quant_max, dtype, axis);
          }
          return result;
        });
  m.def("_xla_dequantize_tensor",
        [](const at::Tensor& input, const std::vector<float>& scale_list,
           const std::vector<int>& zero_point_list, int quant_min,
           int quant_max, const std::string& dtype, int axis) -> at::Tensor {
          at::Tensor result;
          {
            NoGilSection nogil;
            result = DequantizeTensor(input, scale_list, zero_point_list,
                                      quant_min, quant_max, dtype, axis);
          }
          return result;
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
  m.def("_xla_all_gather", [](const at::Tensor& input, int64_t dim,
                              int64_t shard_count, const py::list& groups,
                              bool pin_layout) {
    std::vector<std::vector<int64_t>> replica_groups =
        CreateReduceGroups(groups);
    at::Tensor result;
    {
      NoGilSection nogil;
      result = AllGather(input, dim, shard_count, replica_groups, pin_layout);
    }
    return result;
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
  m.def("_xla_all_gather_coalesced",
        [](const std::vector<at::Tensor>& tensors,
           const std::shared_ptr<torch::lazy::Value>& token, int64_t dim,
           int64_t shard_count, const py::list& groups, bool pin_layout) {
          std::vector<std::vector<int64_t>> replica_groups =
              CreateReduceGroups(groups);
          std::vector<at::Tensor> results;
          std::shared_ptr<torch::lazy::Value> new_token;
          {
            NoGilSection nogil;
            std::tie(results, new_token) = AllGatherCoalesced(
                tensors, token, dim, shard_count, replica_groups, pin_layout);
          }
          auto result_list = py::list(results.size() + 1);
          for (int i = 0; i < results.size(); ++i) {
            result_list[i] = torch::autograd::make_variable(
                results[i], /*requires_grad=*/results[i].requires_grad());
          }
          result_list[results.size()] = new_token;
          return result_list;
        });
  m.def("_xla_all_gather_coalesced_out",
        [](std::vector<at::Tensor>& outputs,
           const std::vector<at::Tensor>& inputs,
           const std::shared_ptr<torch::lazy::Value>& token, int64_t dim,
           int64_t shard_count, const py::list& groups, bool pin_layout) {
          std::vector<std::vector<int64_t>> replica_groups =
              CreateReduceGroups(groups);
          std::shared_ptr<torch::lazy::Value> new_token;
          {
            NoGilSection nogil;
            new_token =
                AllGatherCoalescedOut(outputs, inputs, token, dim, shard_count,
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
  m.def(
      "_xla_spmd_reduce_scatter",
      [](const std::string& reduce_type, const at::Tensor& input, double scale,
         int64_t scatter_dim, int64_t shard_count, const py::list& groups) {
        std::vector<std::vector<int64_t>> replica_groups =
            CreateReduceGroups(groups);
        auto result = tensor_methods::reduce_scatter(
            bridge::GetXlaTensor(input), GetReduceType(reduce_type), scale,
            scatter_dim, shard_count, replica_groups);
        return bridge::AtenFromXlaTensor(std::move(result));
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
  m.def(
      "_xla_reduce_scatter_coalesced",
      [](const std::string& reduce_type, const std::vector<at::Tensor>& inputs,
         const std::shared_ptr<torch::lazy::Value>& token, double scale,
         int64_t scatter_dim, int64_t shard_count, const py::list& groups,
         bool pin_layout) {
        std::vector<std::vector<int64_t>> replica_groups =
            CreateReduceGroups(groups);
        std::vector<at::Tensor> result;
        std::shared_ptr<torch::lazy::Value> new_token;
        {
          NoGilSection nogil;
          std::tie(result, new_token) = ReduceScatterCoalesced(
              reduce_type, inputs, token, scale, scatter_dim, shard_count,
              replica_groups, pin_layout);
        }
        auto result_list = py::list(result.size() + 1);
        for (int i = 0; i < result.size(); ++i) {
          result_list[i] = torch::autograd::make_variable(
              result[i], /*requires_grad=*/result[i].requires_grad());
        }
        result_list[result.size()] = new_token;
        return result_list;
      });
  m.def("_xla_reduce_scatter_coalesced_out",
        [](const std::string& reduce_type, std::vector<at::Tensor>& outputs,
           const std::vector<at::Tensor>& inputs,
           const std::shared_ptr<torch::lazy::Value>& token, double scale,
           int64_t scatter_dim, int64_t shard_count, const py::list& groups,
           bool pin_layout) {
          std::vector<std::vector<int64_t>> replica_groups =
              CreateReduceGroups(groups);
          std::shared_ptr<torch::lazy::Value> new_token;
          {
            NoGilSection nogil;
            new_token = ReduceScatterCoalescedOut(
                reduce_type, outputs, inputs, token, scale, scatter_dim,
                shard_count, replica_groups, pin_layout);
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
        []() { return runtime::GetComputationClient()->GetProcessIndex(); });
  m.def("_xla_get_num_processes",
        []() { return runtime::GetComputationClient()->GetNumProcesses(); });
  m.def("_xla_get_device_ordinal", [](const std::string& device_str) {
    return bridge::AtenDeviceToXlaDevice(device_str).ordinal();
  });
  m.def("_xla_get_device_attributes", [](const std::string& device_str) {
    const absl::flat_hash_map<std::string,
                              runtime::ComputationClient::DeviceAttribute>
        attributes = runtime::GetComputationClient()->GetDeviceAttributes(
            bridge::AtenDeviceToXlaDevice(device_str).toString());

    py::dict dict;
    for (auto const& [name, value] : attributes) {
      dict[py::str(name)] = py::cast(value);
    }
    return dict;
  });
  m.def("_xla_get_all_device_attributes", []() {
    std::vector<std::string> global_devices =
        runtime::GetComputationClient()->GetAllDevices();
    std::vector<py::dict> list;
    for (auto const& device : global_devices) {
      const absl::flat_hash_map<std::string,
                                runtime::ComputationClient::DeviceAttribute>&
          attributes =
              runtime::GetComputationClient()->GetDeviceAttributes(device);
      py::dict dict;
      for (auto const& [name, value] : attributes) {
        dict[py::str(name)] = py::cast(value);
      }
      dict[py::str("name")] = device;
      list.push_back(dict);
    }
    return list;
  });
  m.def(
      "_xla_set_rng_seed",
      [](uint64_t seed, const std::string& device) {
        SetRngSeed(seed, device);
      },
      py::arg("seed") = 101, py::arg("device") = "");
  m.def(
      "_xla_get_rng_seed",
      [](const std::string& device) { return GetRngSeed(device); },
      py::arg("device") = "");
  m.def(
      "_xla_set_should_alias_with_buffer_donor_config",
      [](bool should_alias, const std::string& device_str) {
        torch::lazy::BackendDevice device = GetDeviceOrCurrent(device_str);
        XLAGraphExecutor::Get()->SetAliasWithBufferDonorConfig(should_alias);
      },
      py::arg("should_alias") = false, py::arg("device") = "");
  m.def(
      "_xla_get_should_alias_with_buffer_donor_config",
      [](const std::string& device_str) {
        torch::lazy::BackendDevice device = GetDeviceOrCurrent(device_str);
        return XLAGraphExecutor::Get()->GetAliasWithBufferDonorConfig();
      },
      py::arg("device") = "");
  m.def(
      "_xla_sync_multi",
      [](const std::vector<at::Tensor>& tensors,
         const std::vector<std::string>& devices, bool wait,
         bool sync_xla_data) {
        NoGilSection nogil;
        SyncTensors(tensors, devices, wait, sync_xla_data);
      },
      py::arg("tensors"), py::arg("devices"), py::arg("wait") = true,
      py::arg("sync_xla_data") = true);
  m.def(
      "_xla_warm_up_cache",
      [](const std::vector<at::Tensor>& tensors,
         const std::vector<std::string>& devices) {
        NoGilSection nogil;
        SyncTensors(tensors, devices, /*wait=*/false, /*sync_xla_data=*/false,
                    /*warm_up_cache_only=*/true);
      },
      py::arg("tensors"), py::arg("devices"));
  m.def(
      "_xla_sync_live_tensors",
      [](const std::string& device, const std::vector<std::string>& devices,
         bool wait) {
        NoGilSection nogil;
        SyncLiveTensors(device, devices, wait);
      },
      py::arg("device") = "", py::arg("devices"), py::arg("wait") = true);
  m.def(
      "_xla_step_marker",
      [](const std::string& device, const std::vector<std::string>& devices,
         bool wait, bool reset_scope) {
        NoGilSection nogil;
        StepMarker(device, devices, wait, reset_scope);
      },
      py::arg("device") = "", py::arg("devices"), py::arg("wait") = true,
      py::arg("reset_scope") = true);
  m.def("_get_stablehlo",
        [](const std::vector<at::Tensor>& tensors, const std::string& device,
           const std::vector<std::string>& devices,
           bool emit_bytecode) -> py::bytes {
          NoGilSection nogil;
          EmitMode mode = emit_bytecode ? EmitMode::kStableHloBytecode
                                        : EmitMode::kStableHloReadable;
          std::vector<XLATensorPtr> xtensors;
          if (tensors.empty()) {
            torch::lazy::BackendDevice backend_device =
                GetDeviceOrCurrent(device);
            xtensors = XLAGraphExecutor::Get()->GetLiveTensors(&backend_device);
          } else {
            xtensors = GetXlaTensors(tensors, /*want_all=*/false);
          }
          return py::bytes(
              XLAGraphExecutor::Get()->DumpHloComputation(xtensors, mode));
        });
  m.def("_run_stablehlo",
        [](const std::string& bytecode,
           const std::vector<at::IValue>& graph_inputs)
            -> std::vector<at::Tensor> {
          torch::lazy::BackendDevice device =
              torch_xla::bridge::GetCurrentDevice();
          auto results = XLAGraphExecutor::Get()->ExecuteStablehlo(
              bytecode, graph_inputs, device);
          std::vector<at::Tensor> retlist;
          {
            // Convert result back to at::tensor
            for (const auto& data : results) {
              XLATensorPtr xla_tensor = torch_xla::XLATensor::Create(data);
              retlist.push_back(bridge::AtenFromXlaTensor(xla_tensor));
            }
          }
          return retlist;
        });
  m.def(
      "_xla_wait_device_ops",
      [](const std::vector<std::string>& devices) {
        NoGilSection nogil;
        WaitDeviceOps(devices);
      },
      py::arg("devices"));
  m.def("_get_executed_fallback_ops", []() { return GetFallbackOperations(); });
  m.def("_xla_counter_names", []() {
    auto counter_names = torch::lazy::GetCounterNames();
    auto xla_counter_names = runtime::metrics::GetCounterNames();
    counter_names.insert(counter_names.end(), xla_counter_names.begin(),
                         xla_counter_names.end());
    return counter_names;
  });
  m.def("_xla_counter_value", [](const std::string& name) -> py::object {
    auto* data = torch::lazy::GetCounter(name);
    if (data != nullptr) {
      return py::cast<int64_t>(data->Value());
    }

    auto* xla_data = runtime::metrics::GetCounter(name);
    return xla_data != nullptr ? py::cast<int64_t>(xla_data->Value())
                               : py::none();
  });
  // TORCH_LAZY_COUNTER
  m.def("_xla_increment_counter",
        [](const std::string& name, uint64_t inc_val) {
          torch::lazy::Counter* counter = new ::torch::lazy::Counter(name);
          counter->AddValue(inc_val);
        });
  m.def("_xla_metric_names", []() {
    auto metric_names = torch::lazy::GetMetricNames();
    auto xla_metric_names = runtime::metrics::GetMetricNames();
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
           runtime::metrics_reader::CreateMetricReport(
               runtime::GetComputationClient()->GetMetrics());
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
           runtime::metrics_reader::CreateMetricReport(counter_name_vec,
                                                       metric_name_vec);
  });
  m.def("_clear_xla_counters", []() {
    torch::lazy::MetricsArena::Get()->ResetCounters();
    runtime::metrics::ClearCounters();
  });
  m.def("_clear_xla_metrics", []() {
    torch::lazy::MetricsArena::Get()->ResetMetrics();
    runtime::metrics::ClearMetrics();
  });
  m.def(
      "_xla_tensors_report",
      [](size_t nodes_threshold, const std::string& device) {
        return GetLiveTensorsReport(nodes_threshold, device);
      },
      py::arg("nodes_threshold") = 100, py::arg("device") = "");
  m.def("_xla_memory_info",
        [](const std::string& device) { return GetMemoryInfo(device); });
  m.def("_xla_set_mat_mul_precision", [](const std::string& mat_mul_precision) {
    xla::PrecisionConfig::Precision precision =
        ConsumeValue(xla::StringToPrecision(mat_mul_precision));
    XlaHelpers::set_mat_mul_precision(precision);
  });

  py::class_<xla::XlaBuilder, op_builder::BuilderPtr>(m, "XlaBuilder");
  py::class_<op_builder::Op, op_builder::OpPtr>(m, "XlaOp");
  py::class_<runtime::ComputationClient::Computation,
             runtime::ComputationClient::ComputationPtr>(m, "XlaComputation");
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
    runtime::ComputationClient::ComputationPtr computation;
    {
      NoGilSection nogil;
      computation = CreateComputation(name, root->op);
    }
    return computation;
  });
  m.def("_xla_op_computation_from_module_proto",
        [](const std::string& name, const std::string& module_proto) {
          runtime::ComputationClient::ComputationPtr computation;
          {
            NoGilSection nogil;
            computation = CreateComputationFromProto(name, module_proto);
          }
          return computation;
        });
  m.def("_xla_computation_text",
        [](const runtime::ComputationClient::ComputationPtr& computation) {
          std::string hlo_text;
          {
            NoGilSection nogil;
            hlo_text = ConsumeValue(runtime::util::GetComputationHloText(
                computation->computation()));
          }
          return hlo_text;
        });
  m.def("_xla_op_shape", [](op_builder::OpPtr op) {
    const xla::Shape& shape = ShapeHelper::ShapeOfXlaOp(op->op);
    return op_builder::ShapeToPyShape(shape);
  });
  m.def("_xla_op_builder", [](op_builder::OpPtr op) { return op->builder; });
  m.def("_xla_op_create",
        [](op_builder::BuilderPtr builder, const std::string& opname,
           const std::vector<op_builder::OpPtr>& operands, py::dict args) {
          return op_builder::CreateOp(builder, opname, operands, args);
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
  py::class_<xla::OpSharding>(m, "OpSharding")
      .def(py::init([](const py::list& tile_assignment,
                       const py::list& group_assignment,
                       const py::list& replication_groups, int sharding_type) {
        return ShardingUtil::CreateOpSharding(
            tile_assignment, group_assignment, replication_groups,
            ShardingUtil::ShardingType(sharding_type));
      }));
  m.def("_xla_mark_sharding",
        [](const at::Tensor& input, xla::OpSharding sharding) {
          ShardingUtil::XlaMarkSharding(input, sharding);
        });
  m.def("_mark_manual_sharding",
        [](const at::Tensor& input, xla::OpSharding sharding) {
          XLA_CHECK(IsNonDeviceDataIR(input))
              << "Marking any data tensors as manual is not supported";
          ShardingUtil::XlaMarkSharding(input, sharding);
        });
  m.def("_spmd_full_to_shard_shape", [](const at::Tensor& input) -> at::Tensor {
    XLATensorPtr xtensor = bridge::GetXlaTensor(input);
    auto sharding_spec = xtensor->sharding_spec();
    XLA_CHECK(sharding_spec != nullptr) << "Input tensor is not sharded";

    auto shard_shape = xla::ShapeUtil::MakeShape(
        MakeXlaPrimitiveType(xtensor->dtype(), &(xtensor->GetDevice())),
        ShardingUtil::GetShardShape(sharding_spec));
    auto output = xtensor->CreateFrom(torch_xla::MakeNode<CustomSharding>(
        xtensor->GetIrValue(), shard_shape,
        CustomSharding::Type::kSPMDFullToShardShape));
    output->SetShardingSpec(XLATensor::ShardingSpec(
        xla::HloSharding::Manual().ToProto(), shard_shape));
    return bridge::AtenFromXlaTensor(output);
  });
  m.def(
      "_spmd_shard_to_full_shape",
      [](const at::Tensor& input, const xla::OpSharding& sharding,
         const std::vector<int64_t>& output_shape,
         const py::object& output_dtype) -> at::Tensor {
        XLATensorPtr xtensor = bridge::GetXlaTensor(input);
        auto sharding_spec = xtensor->sharding_spec();
        XLA_CHECK(sharding_spec != nullptr &&
                  sharding_spec->sharding.type() == xla::OpSharding::MANUAL)
            << "Input tensor is not manual sharded";

        auto full_shape = xla::ShapeUtil::MakeShape(
            MakeXlaPrimitiveType(
                reinterpret_cast<THPDtype*>(output_dtype.ptr())->scalar_type,
                &(xtensor->GetDevice())),
            output_shape);
        auto output = xtensor->CreateFrom(torch_xla::MakeNode<CustomSharding>(
            xtensor->GetIrValue(), full_shape,
            CustomSharding::Type::kSPMDShardToFullShape));
        output->SetShardingSpec(XLATensor::ShardingSpec(sharding, full_shape));
        return bridge::AtenFromXlaTensor(output);
      });
  m.def("_xla_mark_sharding_dynamo_custom_op",
        [](const at::Tensor& input, const py::list& tile_assignment,
           const py::list& group_assignment, const py::list& replication_groups,
           int sharding_type) {
          c10::List<at::IntArrayRef> tile_assignment_list =
              c10::List<at::IntArrayRef>();
          for (auto t : tile_assignment) {
            tile_assignment_list.push_back(
                at::IntArrayRef(t.cast<std::vector<int64_t>>()));
          }

          c10::List<at::IntArrayRef> group_assignment_list =
              c10::List<at::IntArrayRef>();
          for (auto t : group_assignment) {
            group_assignment_list.push_back(
                at::IntArrayRef(t.cast<std::vector<int64_t>>()));
          }

          c10::List<at::IntArrayRef> replication_groups_list =
              c10::List<at::IntArrayRef>();
          for (auto t : replication_groups) {
            replication_groups_list.push_back(
                at::IntArrayRef(t.cast<std::vector<int64_t>>()));
          }

          ShardingUtil::XlaMarkShardingDynamoCustomOp(
              input, tile_assignment_list, group_assignment_list,
              replication_groups_list, sharding_type);
        });
  m.def("_xla_clear_sharding", [](const at::Tensor& input) {
    XLATensorPtr xtensor = bridge::GetXlaTensor(input);
    xtensor->ClearShardingSpec();
  });
  m.def("_get_xla_sharding_spec", [](const at::Tensor& input) -> std::string {
    XLATensorPtr xtensor = bridge::GetXlaTensor(input);
    return GetXLAShardingSpec(xtensor);
  });
  m.def("_get_xla_op_sharding",
        [](const at::Tensor& input) -> std::optional<xla::OpSharding> {
          XLATensorPtr xtensor = bridge::GetXlaTensor(input);
          XLATensor::ShardingSpecPtr sharding_spec =
              xtensor ? xtensor->sharding_spec() : nullptr;
          if (sharding_spec != nullptr) {
            return sharding_spec->sharding;
          }
          return std::nullopt;
        });
  m.def("_get_xla_sharding_specs",
        [](const std::vector<at::Tensor>& tensors) -> std::vector<std::string> {
          tsl::profiler::TraceMe activity("_get_xla_sharding_specs",
                                          tsl::profiler::TraceMeLevel::kInfo);
          TORCH_LAZY_TIMED("_get_xla_sharding_specs");
          std::vector<std::string> sharding_specs;
          sharding_specs.reserve(tensors.size());
          for (const at::Tensor& tensor : tensors) {
            sharding_specs.push_back(
                GetXLAShardingSpec(bridge::GetXlaTensor(tensor)));
          }
          return sharding_specs;
        });
  m.def("_get_xla_sharding_type",
        [](const at::Tensor& input) -> std::optional<int> {
          XLATensorPtr xtensor = bridge::GetXlaTensor(input);
          auto sharding_spec = xtensor->sharding_spec();
          if (sharding_spec != nullptr) {
            return ShardingUtil::GetShardingType(sharding_spec->sharding);
          }
          return std::nullopt;
        });
  // Reassemble the CPU shards into a global tensor. A new sharded tensor is
  // created from the local shards with the provided sharding annotation
  // attached. The order of the shards should coincide with the order of
  // devices returned by `torch_xla.runtime.local_runtime_devices()`.
  m.def(
      "_global_tensor_from_cpu_shards",
      [](const std::vector<at::Tensor>& shards, const xla::OpSharding& sharding,
         std::optional<std::vector<int64_t>>& global_shape) -> at::Tensor {
        XLA_CHECK(UseVirtualDevice())
            << "Please enable SPMD via `torch_xla.runtime.use_spmd()`";
        auto local_devices = runtime::GetComputationClient()->GetLocalDevices();
        XLA_CHECK(local_devices.size() == shards.size())
            << "Must specify a shard for each local device";
        XLA_CHECK(!global_shape.has_value() ||
                  global_shape.value().size() == shards[0].sizes().size())
            << "Global shape rank must agree with shard rank: expected rank "
            << shards[0].sizes().size() << ", got "
            << global_shape.value().size();

        if (!global_shape.has_value()) {
          // Set a default value for the global shape based on the sharding
          // type.
          if (sharding.type() == xla::OpSharding::OTHER) {
            // Infer the global shape to be the shard shape scaled by the tiling
            // dimensionality.
            auto tile_shape = sharding.tile_assignment_dimensions();
            global_shape = std::vector<int64_t>();
            for (int dim = 0; dim < shards[0].sizes().size(); ++dim) {
              auto global_dim = tile_shape[dim] * shards[0].sizes()[dim];
              global_shape->push_back(global_dim);
            }
          } else if (sharding.type() == xla::OpSharding::REPLICATED) {
            global_shape = shards[0].sizes().vec();
          } else {
            XLA_ERROR() << "Unsupported OpSharding type: " << sharding.type();
          }
        }

        auto device = GetVirtualDevice();
        auto primitive_type =
            MakeXlaPrimitiveType(shards[0].type().scalarType(), &device);
        xla::Shape tensor_shape = MakeArrayShapeFromDimensions(
            global_shape.value(), /*dynamic_dimensions=*/{}, primitive_type,
            static_cast<XlaDeviceType>(device.type()));
        auto sharding_spec =
            std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);

        // Verify that the shard shape is correct for the global shape and
        // sharding spec.
        auto expected_shard_shape = ShardingUtil::GetShardShape(sharding_spec);
        for (auto shard : shards) {
          XLA_CHECK(shard.sizes() == expected_shard_shape)
              << "Input shard shape must include padding: " << shard.sizes()
              << " vs " << expected_shard_shape;
        }

        auto data_handle = ShardingUtil::CreateShardedData(
            shards, local_devices, sharding_spec);
        XLATensorPtr xla_tensor = XLATensor::Create(std::move(data_handle));
        xla_tensor->SetShardingSpec(*sharding_spec);
        auto tensor = bridge::AtenFromXlaTensor(std::move(xla_tensor));
        return torch::autograd::make_variable(tensor,
                                              shards[0].requires_grad());
      },
      py::arg("shards"), py::arg("sharding"),
      py::arg("global_shape") = py::none());
  // Returns the local shards of the tensor, with values taken from the
  // underlying ComputationClient::GetDataShards. As such, the shards will
  // contain any padding that was applied to ensure they all have the same
  // shape. Note that this padding is _not_ included in the global indices
  // returned by `_get_local_shard_replica_and_indices`.
  // For each input tensor, returns a list of shards and their corresponding
  // device string.
  m.def("_get_local_shards",
        [](const std::vector<at::Tensor>& input)
            -> std::vector<std::vector<std::pair<at::Tensor, std::string>>> {
          std::vector<runtime::ComputationClient::DataPtr> handles;
          std::vector<at::ScalarType> element_types;
          // Find all shard handles for transfer
          for (auto& tensor : input) {
            XLATensorPtr xtensor = bridge::GetXlaTensor(tensor);
            XLA_CHECK(xtensor->GetXlaData() != nullptr)
                << "Shard data is not available";
            XLA_CHECK(xtensor->sharding_spec() != nullptr)
                << "Tensor is not sharded";
            auto handle =
                std::dynamic_pointer_cast<runtime::ComputationClient::Data>(
                    xtensor->GetXlaData());
            std::vector<runtime::ComputationClient::DataPtr> shard_handles =
                runtime::GetComputationClient()->GetDataShards(handle);
            handles.insert(handles.end(), shard_handles.begin(),
                           shard_handles.end());
            element_types.insert(element_types.end(), shard_handles.size(),
                                 MaybeUpcastToHostTorchType(
                                     shard_handles[0]->shape().element_type()));
          }

          std::vector<at::Tensor> cpu_shards =
              XlaDataToTensors(WrapXlaData(handles), element_types);
          // Populate the resulting vector of shards and device strings
          std::vector<std::vector<std::pair<at::Tensor, std::string>>> result;
          int shards_per_tensor =
              runtime::GetComputationClient()->GetLocalDevices().size();
          result.reserve(cpu_shards.size() / shards_per_tensor);
          for (int i = 0; i < cpu_shards.size(); i += shards_per_tensor) {
            std::vector<std::pair<at::Tensor, std::string>> shard_devices;
            for (int shard = 0; shard < shards_per_tensor; ++shard) {
              at::Tensor cpu_shard = cpu_shards[i + shard];
              std::string source_device = handles[i + shard]->device();
              std::pair<at::Tensor, std::string> shard_dev(cpu_shard,
                                                           source_device);
              shard_devices.push_back(shard_dev);
            }
            result.push_back(shard_devices);
          }
          return result;
        });
  // For each input tensors' local shards, returns the tuple:
  //        (replica_id: int, indices: Union[List[Slice], Ellipsis]),
  // where `replica_id` is the replica the shard belongs to and `indices` index
  // into the global tensor. The value of `indices` is either a Python list of
  // slices for each dimension or an Ellipsis object indicating that the tensor
  // is replicated. These indices will not reflect any padding that has been
  // applied to the shards. The order of the returned indices matches the order
  // of the shards returned from `_get_local_shards`.
  m.def(
      "_get_local_shard_replica_and_indices",
      [](const std::vector<at::Tensor>& input_tensors)
          -> std::vector<std::vector<std::pair<int, py::object>>> {
        std::vector<std::vector<std::pair<int, py::object>>> result;
        for (auto& tensor : input_tensors) {
          XLATensorPtr xtensor = bridge::GetXlaTensor(tensor);
          XLA_CHECK(xtensor->sharding_spec() != nullptr)
              << "Tensor is not sharded";
          auto handle =
              std::dynamic_pointer_cast<runtime::ComputationClient::Data>(
                  xtensor->GetXlaData());
          auto shards = runtime::GetComputationClient()->GetDataShards(handle);
          std::vector<std::string> shard_devices;
          for (auto& shard : shards) {
            shard_devices.push_back(shard->device());
          }
          auto sharding_spec = xtensor->sharding_spec();
          auto sharding = xtensor->sharding_spec()->sharding;
          auto shard_shape = ShardingUtil::GetShardShape(sharding_spec);
          auto replica_and_indices =
              ShardingUtil::GetShardReplicaAndIndicesForDevices(
                  shard_shape, tensor.sizes().vec(), sharding, shard_devices);

          // Convert each vector<TensorIndex> to List[py::slice] or py::ellipsis
          std::vector<std::pair<int, py::object>> tensor_ind;
          tensor_ind.reserve(shard_devices.size());
          for (auto& device_replica_and_indices : replica_and_indices) {
            auto& replica_id = device_replica_and_indices.first;
            auto& indices = device_replica_and_indices.second;
            XLA_CHECK(indices.size() > 0)
                << "Unexpected empty shard indices for tensor " << tensor;
            if (indices[0].is_ellipsis()) {
              tensor_ind.push_back(std::make_pair(replica_id, py::ellipsis()));
            } else {
              std::vector<py::object> index_slices;
              for (auto& tensor_index : indices) {
                XLA_CHECK(tensor_index.is_slice())
                    << "Unexpected TensorIndex type: " << tensor_index;
                auto slice = tensor_index.slice();
                ssize_t start = slice.start().expect_int();
                ssize_t stop = slice.stop().expect_int();
                ssize_t step = slice.step().expect_int();
                index_slices.push_back(py::slice(start, stop, step));
              }
              tensor_ind.push_back(
                  std::make_pair(replica_id, py::cast(index_slices)));
            }
          }
          result.push_back(tensor_ind);
        }
        return result;
      });
  // Load a list of local shards into an explicitly-sharded tensor. A shard must
  // be provided for each device.
  m.def("_load_local_shards", [](const at::Tensor& tensor,
                                 std::vector<at::Tensor>& shards,
                                 std::vector<std::string>& devices) {
    XLATensorPtr xtensor = bridge::GetXlaTensor(tensor);
    XLA_CHECK(xtensor->sharding_spec() != nullptr)
        << "Cannot load local shards into a non sharded tensor";
    XLA_CHECK(devices.size() ==
              runtime::GetComputationClient()->GetLocalDevices().size())
        << "Shards must be provided for all local devices";
    auto sharding = xtensor->sharding_spec()->sharding;
    auto sharding_spec = xtensor->sharding_spec();
    XLA_CHECK(sharding.type() != xla::OpSharding::REPLICATED)
        << "Replicated tensor should not be loaded from _load_local_shards - "
           "use copy_";
    auto shard_shape = ShardingUtil::GetShardShape(sharding_spec);
    for (auto shard : shards) {
      XLA_CHECK(shard.sizes() == shard_shape)
          << "Input shard shape must include padding: " << shard.sizes()
          << " vs " << shard_shape;
    }
    auto xla_data =
        ShardingUtil::CreateShardedData(shards, devices, sharding_spec);
    xtensor->SetXlaData(xla_data);
  });
  // Initialize the XlaCoordinator in the runtime if not already initialized.
  m.def(
      "_ensure_xla_coordinator_initialized",
      [](int global_rank, int world_size, std::string master_addr,
         std::string master_port) {
        auto comp_client = runtime::GetComputationClient();
        if (!comp_client->CoordinatorInitialized()) {
          runtime::GetComputationClient()->InitializeCoordinator(
              global_rank, world_size, master_addr, master_port);
        }
      },
      py::arg("global_rank"), py::arg("world_size"), py::arg("master_addr"),
      py::arg("master_port") =
          runtime::XlaCoordinator::kDefaultCoordinatorPort);
  // Create a PreemptionSyncManager for the XlaCoordinator. The
  // PreemptionSyncManager will register a SIGTERM handler as a side effect.
  m.def("_activate_preemption_sync_manager", []() {
    auto comp_client = runtime::GetComputationClient();
    XLA_CHECK(comp_client->CoordinatorInitialized())
        << "Coordinator must be initialized";
    auto& coordinator = comp_client->GetCoordinator();
    coordinator.ActivatePreemptionSyncManager();
  });
  // Deactivate the PreemptionSyncManager in the XlaCoordinator if one is active
  m.def("_deactivate_preemption_sync_manager", []() {
    auto comp_client = runtime::GetComputationClient();
    XLA_CHECK(comp_client->CoordinatorInitialized())
        << "Coordinator must be initialized";
    auto& coordinator = comp_client->GetCoordinator();
    coordinator.DeactivatePreemptionSyncManager();
  });
  // Check whether a sync point has been reached. This method requires that the
  // distributed runtime be initialized and a PreemptionSyncManager activated.
  m.def("_sync_point_reached", [](int step) {
    auto comp_client = runtime::GetComputationClient();
    XLA_CHECK(comp_client->CoordinatorInitialized())
        << "Coordinator must be initialized";
    auto& coordinator = comp_client->GetCoordinator();
    return coordinator.ReachedSyncPoint(step);
  });
  m.def("_is_placecholder", [](at::Tensor& input) {
    XLATensorPtr xtensor = bridge::GetXlaTensor(input);
    return xtensor->CurrentDataHandle() &&
           !xtensor->CurrentDataHandle()->HasValue();
  });
  m.def("_init_xla_lazy_backend", []() {
    MapXlaEnvVarsToLazy();
    InitXlaBackend();
  });
  m.def("_set_ir_debug",
        [](bool ir_debug) { FLAGS_torch_lazy_ir_debug = ir_debug; });
  m.def("_get_ir_debug", []() { return FLAGS_torch_lazy_ir_debug; });
  m.def("_set_xla_all_numbers_special_scalars",
        [](bool all_numbers_special_scalars) {
          FLAGS_torch_lazy_all_numbers_special_scalars =
              all_numbers_special_scalars;
        });
  m.def("_get_xla_all_numbers_special_scalars",
        []() { return FLAGS_torch_lazy_all_numbers_special_scalars; });
  m.def("_set_xla_handle_special_scalars", [](bool handle_special_scalars) {
    FLAGS_torch_lazy_handle_special_scalars = handle_special_scalars;
  });
  m.def("_get_xla_handle_special_scalars",
        []() { return FLAGS_torch_lazy_handle_special_scalars; });
  m.def("_set_xla_enable_device_data_cache", [](bool enable_device_data_cache) {
    FLAGS_torch_lazy_enable_device_data_cache = enable_device_data_cache;
  });
  m.def("_get_xla_enable_device_data_cache",
        []() { return FLAGS_torch_lazy_enable_device_data_cache; });
  m.def("_set_use_eager_mode", [](bool use_eager_mode) {
    XLAGraphExecutor::Get()->SetUseEagerMode(use_eager_mode);
  });
  m.def("_get_use_eager_mode",
        []() { return XLAGraphExecutor::Get()->UseEagerMode(); });
  m.def("_set_allow_execution", [](bool allow_execution) {
    XLAGraphExecutor::Get()->SetAllowExecution(allow_execution);
  });
  m.def("_get_allow_execution",
        []() { return XLAGraphExecutor::Get()->AllowExecution(); });
  m.def("_set_current_graph_name", [](std::string current_graph_name) {
    XLAGraphExecutor::Get()->SetCurrentGraphName(current_graph_name);
  });
  m.def("_get_current_graph_name",
        []() { return XLAGraphExecutor::Get()->CurrentGraphName(); });
  m.def("_dynamic_shape_detector_start_session",
        [](const std::string& session) {
          DynamicShapeDetector::Get()->StartSession(session);
        });
  m.def("_dynamic_shape_detector_end_session",
        []() { return DynamicShapeDetector::Get()->EndSession(); });
  m.def("_dynamic_shape_detector_remove_session",
        [](const std::string& session) {
          DynamicShapeDetector::Get()->RemoveSessionIfExists(session);
        });
  m.def("_dynamic_shape_detector_set_max_allowed_traces",
        [](int64_t max_allowed_traces) {
          DynamicShapeDetector::SetMaxAllowedTraces(max_allowed_traces);
        });
  m.def("_dynamic_shape_detector_get_max_allowed_traces",
        [](int64_t max_allowed_traces) {
          return DynamicShapeDetector::GetMaxAllowedTraces();
        });
  m.def("_replace_xla_tensor",
        [](at::Tensor& self, const at::Tensor& source) -> at::Tensor& {
          return XLANativeFunctions::set_(self, source);
        });
  m.def("_xla_custom_call",
        [](const std::vector<at::Tensor>& inputs, const std::string& target,
           const std::vector<std::vector<int64_t>>& output_shapes,
           const std::vector<py::object>& output_dtypes, bool has_side_effect,
           const std::string& backend_config, const int api_version,
           const std::unordered_map<std::string, std::string>&
               frontend_attributes) -> std::vector<at::Tensor> {
          std::vector<at::ScalarType> dtypes;
          dtypes.reserve(output_dtypes.size());
          for (auto& dtype : output_dtypes) {
            dtypes.push_back(
                reinterpret_cast<THPDtype*>(dtype.ptr())->scalar_type);
          }

          auto xtensors = tensor_methods::custom_call(
              bridge::GetXlaTensors(inputs), target, output_shapes, dtypes,
              has_side_effect, backend_config, api_version,
              frontend_attributes);
          return bridge::AtenFromXlaTensors(std::move(xtensors));
        });
  m.def("_xla_tpu_custom_call",
        [](const std::vector<at::Tensor>& inputs, const std::string& payload,
           const std::vector<std::vector<int64_t>>& output_shapes,
           const std::vector<py::object>& output_dtypes)
            -> std::vector<at::Tensor> {
          return XlaCustomCall(inputs, payload, output_shapes, output_dtypes,
                               /*is_tpu=*/true);
        });
  m.def("_has_cuda_support", []() {
#ifdef GOOGLE_CUDA
    return true;
#else
    return false;
#endif
  });
  m.def("_xla_gpu_custom_call",
        [](const std::vector<at::Tensor>& inputs, const std::string& payload,
           const std::vector<std::vector<int64_t>>& output_shapes,
           const std::vector<py::object>& output_dtypes)
            -> std::vector<at::Tensor> {
          return XlaCustomCall(inputs, payload, output_shapes, output_dtypes,
                               /*is_tpu=*/false);
        });
  m.def("_xla_register_custom_call_target",
        [](const std::string& fn_name, const py::capsule& function_ptr,
           const std::string& platform) {
          runtime::GetComputationClient()->RegisterCustomCall(
              fn_name, function_ptr.get_pointer(), platform);
        });
  m.def("_set_xla_custom_op_name_prefix",
        [](const at::Tensor& input, const std::string& op_name_prefix,
           size_t max_call_stack_depth) -> bool {
          XLATensorPtr xtensor = bridge::GetXlaTensor(input);
          std::shared_ptr<torch::lazy::UserMetaData> user_meta =
              std::make_shared<CustomOpNameMetaData>(op_name_prefix,
                                                     max_call_stack_depth);
          return xtensor->SetNodeUserMetadata(user_meta);
        });
  m.def("_get_all_reduce_token",
        [](const std::string& device_str) -> const torch::lazy::Value& {
          auto device = GetDeviceOrCurrent(device_str);
          return GetAllReduceToken(device);
        });
  m.def("_set_all_reduce_token",
        [](const std::string& device_str,
           const std::shared_ptr<torch::lazy::Value>& token) {
          auto device = GetDeviceOrCurrent(device_str);
          SetAllReduceToken(device, token);
        });

  BuildProfilerSubmodule(&m);
  BuildLoweringContextSubmodule(&m);

  m.def("_get_tensors_handle",
        [](const std::vector<at::Tensor>& tensors) -> std::vector<int64_t> {
          std::vector<torch::lazy::BackendData::Handle> handles;
          handles.reserve(tensors.size());
          for (auto& tensor : tensors) {
            handles.push_back(bridge::GetXlaTensor(tensor)->GetHandle());
          }
          return handles;
        });
  m.def("_xla_mark_tensor",
        [](const at::Tensor& input, const std::string& info) {
          TORCH_LAZY_COUNTER("XlaMarkTensor", 1);
          at::Tensor result;
          {
            NoGilSection nogil;
            result = MarkTensor(input, info);
          }
          return result;
        });
  m.def("_xla_mark_dynamic", [](const at::Tensor& input, uint32_t dim) {
    TORCH_LAZY_COUNTER("XlaMarkDynamic", 1);
    XLATensorPtr xtensor = bridge::GetXlaTensor(input);
    xtensor->MarkDynamicDimension(dim);
  });
  m.def("_xla_dynamic_expand",
        [](const at::Tensor& input, const std::vector<int64_t>& size,
           const at::Tensor& src_tensor, int src_dim,
           int target_dim) -> at::Tensor {
          at::Tensor result;
          {
            NoGilSection nogil;
            result =
                DynamicExpand(input, size, src_tensor, src_dim, target_dim);
          }
          return result;
        });
  m.def("_xla_dynamic_view",
        [](const at::Tensor& input, const std::vector<int64_t>& size,
           const at::Tensor& src_tensor, int src_dim, int target_dim,
           float mul_scaler) -> at::Tensor {
          at::Tensor result;
          {
            NoGilSection nogil;
            result = DynamicView(input, size, src_tensor, src_dim, target_dim,
                                 mul_scaler);
          }
          return result;
        });

  // This api will set the `should_donate_buffer_` field in the
  // ComputationClient::Data. This api is currently only useful if you are
  // running with `torch.compile`. Buffer assocaited with data with
  // `should_donate_buffer_` set to true will be donated to the output, You
  // should only use this api if
  // 1. You are using torch.compile
  // 2. You will inplace update a tensor in the `torch.compiled` function(so the
  //    currnet buffer can be donated after compuation)
  m.def("_set_buffer_donation",
        [](at::Tensor& input, bool should_donate) -> bool {
          XLATensorPtr xtensor = bridge::GetXlaTensor(input);
          bool buffer_donation_updated = false;
          if (!xtensor) {
            // input tensor is not a XLATensor, return here.
          } else if (xtensor->CurrentDataHandle() != nullptr) {
            auto data =
                std::dynamic_pointer_cast<runtime::ComputationClient::Data>(
                    xtensor->CurrentDataHandle());
            data->set_should_donate_buffer(should_donate);
            buffer_donation_updated = true;
          } else if (xtensor->CurrentIrValue().node != nullptr) {
            torch::lazy::NodePtr node = xtensor->CurrentIrValue().node;
            auto device_data = torch_xla::DeviceData::Cast(node.get());
            if (device_data != nullptr) {
              device_data->set_buffer_donation(should_donate);
              buffer_donation_updated = true;
            }
          }
          if (buffer_donation_updated) {
            TORCH_LAZY_COUNTER("XlaSetBufferDonation", 1);
          }
          return buffer_donation_updated;
        });

  m.def("_get_buffer_donation", [](const at::Tensor& input) -> bool {
    XLATensorPtr xtensor = bridge::GetXlaTensor(input);
    if (!xtensor) {
      return false;
    } else if (xtensor->CurrentDataHandle() != nullptr) {
      auto data = std::dynamic_pointer_cast<runtime::ComputationClient::Data>(
          xtensor->CurrentDataHandle());
      return data->should_donate_buffer();
    } else if (xtensor->CurrentIrValue().node != nullptr) {
      auto device_data =
          torch_xla::DeviceData::Cast(xtensor->CurrentIrValue().node.get());
      if (device_data != nullptr) {
        return device_data->get_buffer_donation();
      } else {
        return false;
      }
    }
    return false;
  });

  m.def("_on_ready_callback",
        [](const at::Tensor& tensor, const std::function<void()>& callback) {
          XLATensorPtr xtensor = bridge::GetXlaTensor(tensor);
          XLA_CHECK(xtensor) << "The input is not an XLA tensor.";
          // Wait for placeholder `Data`s to be assigned
          XLAGraphExecutor::Get()->WaitDeviceOps({});
          std::shared_ptr<runtime::ComputationClient::Data> data;
          if (xtensor->CurrentDataHandle() != nullptr) {
            data = UnwrapXlaData(xtensor->CurrentDataHandle());
          } else if (xtensor->CurrentIrValue().node != nullptr) {
            DeviceData* device_data =
                DeviceData::Cast(xtensor->CurrentIrValue().node.get());
            if (device_data != nullptr) {
              data = UnwrapXlaData(device_data->data());
            } else {
              XLA_ERROR() << "Could not get the buffer pointer for XLATensor "
                             "with IR that's not DeviceData";
            }
            XLA_ERROR() << "Could not get buffer for tensor";
          }
          runtime::GetComputationClient()->OnReadyCallback(data, callback);
        });

  m.def("_unsafe_buffer_pointer",
        [](const at::Tensor& input) -> std::uintptr_t {
          XLATensorPtr xtensor = bridge::GetXlaTensor(input);
          XLA_CHECK(xtensor) << "The input is not an XLA tensor.";
          if (xtensor->CurrentDataHandle() != nullptr) {
            std::shared_ptr<runtime::ComputationClient::Data> data =
                std::dynamic_pointer_cast<runtime::ComputationClient::Data>(
                    xtensor->CurrentDataHandle());
            return runtime::GetComputationClient()->UnsafeBufferPointer(data);
          } else if (xtensor->CurrentIrValue().node != nullptr) {
            DeviceData* device_data =
                DeviceData::Cast(xtensor->CurrentIrValue().node.get());
            if (device_data != nullptr) {
              torch::lazy::BackendDataPtr data = device_data->data();
              return runtime::GetComputationClient()->UnsafeBufferPointer(
                  UnwrapXlaData(data));
            } else {
              XLA_ERROR() << "Could not get the buffer pointer for XLATensor "
                             "with IR that's not DeviceData";
            }
          }
          XLA_ERROR() << "Could not get the buffer pointer for XLATensor "
                         "without a data handle or an IR.";
        });

  // from an XLA tensor to a PyCapsule.
  // When consuming the PyCapsule, we should synchronize
  // (waits for all kernels in all streams on a CUDA device to complete) if
  // the current stream is different from the ext_data's stream. Otherwise, we
  // may risk of getting incorrect results.
  m.def("_to_dlpack", [](const at::Tensor& input) -> py::handle {
    DLManagedTensor* dlMTensor;
    {
      NoGilSection nogil;
      dlMTensor = torch_xla::toDLPack(input);
    }
    return PyCapsule_New(dlMTensor, "dltensor", dlPack_Capsule_Destructor);
  });

  // from a dlpack PyCapsule to an XLA tensor
  // If ext_data is the result of an CUDA computation, we should synchronize
  // (waits for all kernels in all streams on a CUDA device to complete) if
  // the current stream is different from the ext_data's stream. Otherwise, we
  // may risk of getting incorrect results. Or you can use torch_xla's
  // from_dlpack(cuda_tensor) and it will handle the synchronization for you.
  m.def("_from_dlpack", [](py::handle ext_data) -> at::Tensor {
    return tensor_fromDLPack(ext_data.ptr());
  });

  // -------------Dynamo Integration API Start-------------------------
  /*
   * Return tensor ids and at::tensors for all DeviceData nodes that is needed
   * to compute the value of tensors.
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
          std::unordered_set<torch::lazy::BackendData::Handle> data_handles;

          for (const torch::lazy::Node* nodeptr : post_order) {
            const auto backend_data =
                torch::lazy::getBackend()->GetComputationDataFromNode(nodeptr);
            if (!backend_data) {
              continue;
            }

            // Dedup by handle
            torch::lazy::BackendData::Handle handle = backend_data->GetHandle();
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
    auto xla_device = GetDeviceOrCurrent(device);
    SetAllReduceToken(xla_device, nullptr);
  });

  m.def(
      "_unique_id_for_ir_and_data",
      [](const at::Tensor& tensor) -> std::string {
        XLATensorPtr xtensor = bridge::GetXlaTensor(tensor);
        if (xtensor->CurrentIrValue()) {
          torch::lazy::Value value = xtensor->CurrentIrValue();
          return std::to_string((uintptr_t)value.node.get()) + ", " +
                 std::to_string(value.index);
        } else if (xtensor->CurrentDataHandle()) {
          return std::to_string((uintptr_t)xtensor->CurrentDataHandle().get());
        } else {
          return std::to_string((uintptr_t)xtensor.get());
        }
      });

  m.def("_run_cached_graph",
        [](const std::string& hash_str,
           const std::vector<at::IValue>& graph_inputs)
            -> std::vector<at::Tensor> {
          XLA_CHECK(hash_str.size() == sizeof(torch::lazy::hash_t));
          torch::lazy::hash_t hash = *(torch::lazy::hash_t*)(hash_str.c_str());
          // Device will be Virtual device if SPMD is enabled.
          torch::lazy::BackendDevice device =
              torch_xla::bridge::GetCurrentDevice();
          auto results = XLAGraphExecutor::Get()->ExecuteComputationWithBarrier(
              hash, graph_inputs, device);
          std::vector<at::Tensor> retlist;
          {
            TORCH_LAZY_TIMED("RunCachedGraphOutputData");
            // Convert result back to at::tensor
            for (const auto& data : results) {
              XLATensorPtr xla_tensor = torch_xla::XLATensor::Create(data);
              retlist.push_back(bridge::AtenFromXlaTensor(xla_tensor));
            }
          }

          return retlist;
        });
  // -------------Dynamo Integration API End-------------------------
  m.def(
      "_register_pjrt_plugin",
      [](std::string name, std::shared_ptr<const runtime::PjRtPlugin> plugin) {
        runtime::RegisterPjRtPlugin(name, plugin);
      });
  py::class_<runtime::PjRtPlugin, PyPjRtPlugin,
             std::shared_ptr<runtime::PjRtPlugin>>(m, "PjRtPlugin")
      .def(py::init<>())
      .def("library_path", &runtime::PjRtPlugin::library_path)
      .def("client_create_options", &runtime::PjRtPlugin::client_create_options)
      .def("requires_xla_coordinator",
           &runtime::PjRtPlugin::requires_xla_coordinator);
}
}  // namespace

void InitXlaBindings(py::module m) { InitXlaModuleBindings(m); }

}  // namespace torch_xla

PYBIND11_MODULE(_XLAC, m) { torch_xla::InitXlaBindings(m); }
