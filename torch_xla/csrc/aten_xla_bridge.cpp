#include "torch_xla/csrc/aten_xla_bridge.h"

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/tensor_impl.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace bridge {
namespace {

c10::optional<XLATensor> TryGetXlaTensor(const at::Tensor& tensor) {
  XLATensorImpl* impl =
      dynamic_cast<XLATensorImpl*>(tensor.unsafeGetTensorImpl());
  if (impl == nullptr) {
    return c10::nullopt;
  }
  return impl->tensor();
}

}  // namespace

XLATensor GetXlaTensor(const at::Tensor& tensor) {
  auto xtensor = TryGetXlaTensor(tensor);
  XLA_CHECK(xtensor) << "Input tensor is not an XLA tensor: "
                     << tensor.toString();
  return *xtensor;
}

std::vector<XLATensor> GetXlaTensors(
    tensorflow::gtl::ArraySlice<const at::Tensor> tensors) {
  std::vector<XLATensor> xla_tensors;
  xla_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    xla_tensors.push_back(bridge::GetXlaTensor(tensor));
  }
  return xla_tensors;
}

XLATensor GetXlaTensorUnwrap(const at::Tensor& tensor) {
  return GetXlaTensor(ToTensor(tensor));
}

XLATensor GetOrCreateXlaTensor(const at::Tensor& tensor, const Device& device) {
  auto xtensor = TryGetXlaTensor(tensor);
  return xtensor ? *xtensor
                 : XLATensor::Create(tensor, device, /*requires_grad=*/false);
}

std::vector<at::Tensor> XlaCreateTensorList(
    const at::TensorList& tensors, const std::vector<bool>* writeable) {
  std::vector<at::Tensor> aten_xla_tensors(tensors.size());
  std::vector<XLATensor> xla_tensors;
  // We need to separate out the defined tensors first, GetXlaTensor() doesn't
  // work with undefined tensors.
  std::vector<bool> defined_writeable;
  std::vector<bool> to_translate(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    const at::Tensor& tensor = tensors[i];
    if (!tensor.defined()) {
      XLA_CHECK(writeable == nullptr || !(*writeable)[i])
          << "Trying to write to an undefined tensor";
    } else if (tensor.device().is_cpu()) {
      aten_xla_tensors[i] = ToTensor(tensor);
    } else {
      to_translate[i] = true;
      xla_tensors.push_back(GetXlaTensorUnwrap(tensor));
      if (writeable != nullptr) {
        defined_writeable.push_back((*writeable)[i]);
      }
    }
  }
  auto defined_aten_xla_tensors = XLATensor::GetTensors(
      &xla_tensors, writeable ? &defined_writeable : nullptr);
  // Insert undefined tensors into the result, back into the original undefined
  // positions.
  for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
    if (to_translate[i]) {
      aten_xla_tensors[i] = std::move(defined_aten_xla_tensors[defined_pos++]);
    }
  }
  return aten_xla_tensors;
}

c10::optional<Device> GetXlaDevice(const at::Tensor& tensor) {
  auto xtensor = TryGetXlaTensor(ToTensor(tensor));
  if (!xtensor) {
    return c10::nullopt;
  }
  return xtensor->GetDevice();
}

c10::optional<Device> GetXlaDevice(const at::TensorList& tensors) {
  for (const auto& tensor : tensors) {
    auto device = GetXlaDevice(tensor);
    if (device) {
      return device;
    }
  }
  return c10::nullopt;
}

c10::optional<Device> GetXlaDevice(const at::TensorOptions& tensor_options) {
  if (!tensor_options.has_device()) {
    return c10::nullopt;
  }
  return GetXlaDevice(tensor_options.device());
}

c10::optional<Device> GetXlaDevice(const c10::Device& device) {
  if (device.type() != at::kXLA) {
    return c10::nullopt;
  }
  Device xla_device = *GetDefaultDevice();
  xla_device.ordinal = device.has_index() ? device.index() : 0;
  return xla_device;
}

Device AtenDeviceToXlaDevice(const c10::Device& device) {
  int ordinal = device.has_index() ? device.index() : 0;
  switch (device.type()) {
    case at::kCPU:
      return Device(DeviceType::CPU, ordinal);
    case at::kCUDA:
      return Device(DeviceType::GPU, ordinal);
    case at::kXLA: {
      Device xla_device = *GetDefaultDevice();
      xla_device.ordinal = ordinal;
      return xla_device;
    }
    default:
      XLA_ERROR() << "Device type " << DeviceTypeName(device.type(), false)
                  << " not supported";
  }
}

at::Tensor AtenFromXlaTensor(XLATensor xla_tensor) {
  return at::Tensor(c10::make_intrusive<XLATensorImpl>(std::move(xla_tensor)));
}

std::vector<at::Tensor> AtenFromXlaTensors(
    tensorflow::gtl::ArraySlice<const XLATensor> xla_tensors) {
  std::vector<at::Tensor> tensors;
  tensors.reserve(xla_tensors.size());
  for (auto& tensor : xla_tensors) {
    tensors.emplace_back(AtenFromXlaTensor(tensor));
  }
  return tensors;
}

at::Tensor CreateXlaTensor(at::Tensor tensor,
                           const c10::optional<Device>& device) {
  if (tensor.defined() && device) {
    XLATensor xla_tensor =
        XLATensor::Create(std::move(tensor), *device, /*requires_grad=*/false);
    tensor = AtenFromXlaTensor(xla_tensor);
  }
  return tensor;
}

std::vector<at::Tensor> CreateXlaTensors(const std::vector<at::Tensor>& tensors,
                                         const c10::optional<Device>& device) {
  std::vector<at::Tensor> xtensors;
  for (auto& tensor : tensors) {
    xtensors.push_back(CreateXlaTensor(tensor, device));
  }
  return xtensors;
}

}  // namespace bridge
}  // namespace torch_xla
