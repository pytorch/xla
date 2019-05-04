#include "torch_xla/csrc/init_python_bindings.h"

#include <c10/core/Device.h>
#include <c10/util/Optional.h>

#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/variable.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/aten_xla_type.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/ir_util.h"
#include "torch_xla/csrc/python_util.h"
#include "torch_xla/csrc/tensor_impl.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

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

std::string GetTensorsDump(
    const std::vector<at::Tensor>& tensors,
    const std::function<std::string(
        tensorflow::gtl::ArraySlice<const ir::Node* const>)>& coverter) {
  std::vector<const ir::Node*> nodes;
  std::vector<ir::Value> values;
  for (auto& tensor : tensors) {
    XLATensor xtensor = bridge::GetXlaTensor(ToTensor(tensor));
    values.push_back(xtensor.GetIrValue());
    nodes.push_back(values.back().node.get());
  }
  return coverter(nodes);
}

std::string SetCurrentDevice(const std::string& device_str) {
  c10::Device prev_device =
      XLATensorImpl::SetCurrentAtenDevice(c10::Device(device_str));
  std::stringstream ss;
  ss << prev_device;
  return ss.str();
}

std::string GetCurrentDevice() {
  std::stringstream ss;
  ss << XLATensorImpl::GetCurrentAtenDevice();
  return ss.str();
}

void SetReplicationDevices(const std::vector<std::string>& devices) {
  std::vector<std::string> replication_devices;
  for (auto& device_str : devices) {
    Device device = bridge::AtenDeviceToXlaDevice(c10::Device(device_str));
    replication_devices.emplace_back(device.ToString());
  }
  xla::ComputationClient::Get()->SetReplicationDevices(replication_devices);
}

std::vector<std::string> GetReplicationDevices() {
  std::vector<std::string> replication_devices;
  for (auto& device_str :
       xla::ComputationClient::Get()->GetReplicationDevices()) {
    c10::Device device = bridge::XlaDeviceToAtenDevice(Device(device_str));
    replication_devices.emplace_back(bridge::ToXlaString(device));
  }
  return replication_devices;
}

void InsertCrossReplicaSum(const std::vector<at::Tensor>& tensors, double scale,
                           const py::list& groups) {
  std::vector<std::vector<xla::int64>> crs_groups;
  for (auto& group : groups) {
    crs_groups.emplace_back();
    for (auto& replica_id : group.cast<py::list>()) {
      crs_groups.back().push_back(replica_id.cast<xla::int64>());
    }
  }
  for (auto& tensor : tensors) {
    XLATensor xtensor = bridge::GetXlaTensor(ToTensor(tensor));
    XLATensor::cross_replica_sum_(xtensor, scale, crs_groups);
  }
}

void SyncTensors(const std::vector<at::Tensor>& tensors, bool wait) {
  std::vector<XLATensor> xtensors;
  for (auto& tensor : tensors) {
    auto xtensor = bridge::TryGetXlaTensor(ToTensor(tensor));
    if (xtensor) {
      xtensors.push_back(*xtensor);
    }
  }
  XLATensor::SyncTensorsGraph(&xtensors, wait);
}

void SyncLiveTensors(const std::string& device_str) {
  auto opt_device = GetOptionalDevice(device_str);
  XLATensor::SyncLiveTensorsGraph(opt_device ? &opt_device.value() : nullptr);
}

void StepMarker(const std::string& device_str) {
  auto opt_device = GetOptionalDevice(device_str);
  const Device* device = opt_device ? &opt_device.value() : nullptr;
  XLATensor::SyncLiveTensorsGraph(device);
  XLATensor::MarkStep(device);
}

std::string GetTensorsHloGraph(const std::vector<at::Tensor>& tensors) {
  std::vector<XLATensor> xtensors;
  for (auto& tensor : tensors) {
    auto xtensor = bridge::TryGetXlaTensor(ToTensor(tensor));
    if (xtensor) {
      xtensors.push_back(*xtensor);
    }
  }
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
      auto post_order = ir::Util::ComputePostOrder({ir_value.node.get()});
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
        ss << ir::DumpUtil::PostOrderToText(post_order);
        ss << "\n\n";
      }
    }
  }
  return ss.str();
}

std::vector<at::Tensor> GetXlaTensorsFromAten(
    const std::vector<at::Tensor>& aten_tensors,
    const std::vector<std::string>& devices) {
  std::vector<std::string> xla_devices;
  xla_devices.reserve(devices.size());
  for (auto& device_str : devices) {
    Device device = bridge::AtenDeviceToXlaDevice(c10::Device(device_str));
    xla_devices.emplace_back(device.ToString());
  }
  std::vector<at::Tensor> tensors;
  tensors.reserve(aten_tensors.size());
  for (auto& aten_tensor : aten_tensors) {
    tensors.push_back(ToTensor(aten_tensor));
  }

  auto data_handles = CreateTensorsData(tensors, xla_devices);

  std::vector<at::Tensor> xla_tensors;
  xla_tensors.reserve(data_handles.size());
  for (auto& data_handle : data_handles) {
    XLATensor xla_tensor = XLATensor::Create(std::move(data_handle));
    xla_tensors.push_back(bridge::AtenFromXlaTensor(std::move(xla_tensor)));
  }
  return xla_tensors;
}

void InitXlaModuleBindings(py::module m) {
  m.def("_initialize_aten_bindings",
        []() { AtenXlaType::InitializeAtenBindings(); });
  m.def("_get_xla_tensor", [](const at::Tensor& tensor) -> XLATensor {
    return bridge::GetXlaTensor(ToTensor(tensor));
  });
  m.def("_get_xla_tensors_dot",
        [](const std::vector<at::Tensor>& tensors) -> std::string {
          auto coverter =
              [](tensorflow::gtl::ArraySlice<const ir::Node* const> nodes) {
                return ir::DumpUtil::ToDot(nodes);
              };
          return GetTensorsDump(tensors, coverter);
        });
  m.def("_get_xla_tensors_text",
        [](const std::vector<at::Tensor>& tensors) -> std::string {
          auto coverter =
              [](tensorflow::gtl::ArraySlice<const ir::Node* const> nodes) {
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
      for (auto& tensor : xla_tensors) {
        result.push_back(torch::autograd::make_variable(tensor));
      }
    }
    return result;
  });
  m.def("_xla_get_devices",
        []() { return xla::ComputationClient::Get()->GetAvailableDevices(); });
  m.def("_xla_set_replication_devices",
        [](const std::vector<std::string>& devices) {
          NoGilSection nogil;
          SetReplicationDevices(devices);
        });
  m.def("_xla_get_replication_devices", []() {
    NoGilSection nogil;
    return GetReplicationDevices();
  });
  m.def("_xla_replication_device_count", []() {
    return xla::ComputationClient::Get()->GetReplicationDevices().size();
  });
  m.def("_xla_cross_replica_sum", [](const std::vector<at::Tensor>& tensors,
                                     double scale, const py::list& groups) {
    NoGilSection nogil;
    InsertCrossReplicaSum(tensors, scale, groups);
  });
  m.def("_xla_set_default_device",
        [](const std::string& device) { return SetCurrentDevice(device); });
  m.def("_xla_get_default_device", []() { return GetCurrentDevice(); });
  m.def(
      "_xla_sync_multi",
      [](const std::vector<at::Tensor>& tensors, bool wait) {
        NoGilSection nogil;
        SyncTensors(tensors, wait);
      },
      py::arg("tensors"), py::arg("wait") = true);
  m.def(
      "_xla_sync_live_tensors",
      [](const std::string& device) {
        NoGilSection nogil;
        SyncLiveTensors(device);
      },
      py::arg("device") = "");
  m.def(
      "_xla_step_marker",
      [](const std::string& device) {
        NoGilSection nogil;
        StepMarker(device);
      },
      py::arg("device") = "");
  m.def("_xla_counter_value", [](const std::string& name) -> py::object {
    xla::metrics::CounterData* data = xla::metrics::GetCounter(name);
    return data != nullptr ? py::cast<int64_t>(data->Value()) : py::none();
  });
  m.def("_xla_metrics_report",
        []() { return xla::metrics::CreateMetricReport(); });
  m.def(
      "_xla_tensors_report",
      [](size_t nodes_threshold, const std::string& device) {
        return GetLiveTensorsReport(nodes_threshold, device);
      },
      py::arg("nodes_threshold") = 100, py::arg("device") = "");
  m.def(
      "_xla_set_use_full_mat_mul_precision",
      [](bool use_full_mat_mul_precision) {
        XlaHelpers::set_mat_mul_precision(use_full_mat_mul_precision
                                              ? xla::PrecisionConfig::HIGHEST
                                              : xla::PrecisionConfig::DEFAULT);
      },
      py::arg("use_full_mat_mul_precision") = true);
}

}  // namespace

void InitXlaBindings(py::module m) { InitXlaModuleBindings(m); }

}  // namespace torch_xla

PYBIND11_MODULE(_XLAC, m) { torch_xla::InitXlaBindings(m); }
