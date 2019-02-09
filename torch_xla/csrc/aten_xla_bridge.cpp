#include "aten_xla_bridge.h"

#include "device.h"
#include "tensor_impl.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_util.h"

namespace torch_xla {
namespace bridge {

XLATensor& GetXlaTensor(const at::Tensor& tensor) {
  XLATensorImpl* impl =
      dynamic_cast<XLATensorImpl*>(tensor.unsafeGetTensorImpl());
  XLA_CHECK(impl != nullptr)
      << "Input tensor is not an XLA tensor: " << tensor.toString();
  return impl->tensor();
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
      xla_tensors.push_back(GetXlaTensor(ToTensor(tensor)));
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

std::vector<at::Tensor> CreateXlaTensors(const std::vector<at::Tensor>& tensors,
                                         const Device& device) {
  std::vector<at::Tensor> xtensors;
  for (auto& tensor : tensors) {
    xtensors.push_back(CreateXlaTensor(tensor, device));
  }
  return xtensors;
}

Device XlaTensorDevice(const at::Tensor& tensor) {
  return GetXlaTensor(ToTensor(tensor)).GetDevice();
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

Device XlaTensorDevice(const at::TensorOptions& tensor_options) {
  return tensor_options.has_device()
             ? AtenDeviceToXlaDevice(tensor_options.device())
             : *GetDefaultDevice();
}

at::Tensor AtenFromXlaTensor(XLATensor xla_tensor) {
  return at::Tensor(c10::make_intrusive<XLATensorImpl>(std::move(xla_tensor)));
}

at::Tensor CreateXlaTensor(at::Tensor tensor, const Device& device) {
  if (tensor.defined()) {
    XLATensor xla_tensor =
        XLATensor::Create(std::move(tensor), device, /*requires_grad=*/false);
    tensor = AtenFromXlaTensor(xla_tensor);
  }
  return tensor;
}

}  // namespace bridge
}  // namespace torch_xla
