#include "aten_xla_bridge.h"

#include "tensor_impl.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace bridge {

at::Tensor CreateEmptyTensor(at::IntList size,
                             const at::TensorOptions& options) {
  return at::empty(size, options.device(at::kCPU));
}

at::Tensor CreateRandTensor(at::IntArrayRef size,
                            at::Generator* generator,
                            const at::TensorOptions& options) {
  return at::randn(size, generator, options.device(at::DeviceType::CPU));
}

at::Tensor CreateRandTensor(at::IntArrayRef size,
                            const at::TensorOptions& options) {
  return at::randn(size, options.device(at::DeviceType::CPU));
}

XLATensor& GetXlaTensor(const at::Tensor& tensor) {
  XLATensorImpl* impl =
      dynamic_cast<XLATensorImpl*>(tensor.unsafeGetTensorImpl());
  XLA_CHECK(impl != nullptr)
      << "Input tensor is not an XLA tensor: " << tensor.toString();
  return impl->tensor();
}

std::vector<at::Tensor> XlaCreateTensorList(
    const at::TensorList& tensors, const std::vector<bool>* writeable) {
  std::vector<XLATensor> xla_tensors;
  // We need to separate out the defined tensors first, GetXlaTensor() doesn't
  // work with undefined tensors.
  std::vector<bool> defined_writeable;
  std::vector<bool> tensor_is_defined(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    const at::Tensor& tensor = tensors[i];
    if (!tensor.defined()) {
      XLA_CHECK(writeable == nullptr || !(*writeable)[i])
          << "Trying to write to an undefined tensor";
      continue;
    }
    tensor_is_defined[i] = true;
    xla_tensors.push_back(GetXlaTensor(tensor));
    if (writeable != nullptr) {
      defined_writeable.push_back((*writeable)[i]);
    }
  }
  auto defined_aten_xla_tensors = XLATensor::GetTensors(
      &xla_tensors, writeable ? &defined_writeable : nullptr);
  // Insert undefined tensors into the result, back into the original undefined
  // positions.
  std::vector<at::Tensor> aten_xla_tensors;
  for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
    if (tensor_is_defined[i]) {
      aten_xla_tensors.push_back(
          std::move(defined_aten_xla_tensors[defined_pos++]));
    } else {
      aten_xla_tensors.emplace_back();
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
  return GetXlaTensor(tensor).GetDevice();
}

Device XlaTensorDevice(const at::TensorOptions& tensor_options) {
  static Device* xla_device =
      new Device(xla::ComputationClient::Get()->GetDefaultDevice());
  at::DeviceType at_device_type = tensor_options.device().type();
  switch (at_device_type) {
    case at::kCPU:
      return Device(DeviceType::CPU, 0);
    case at::kCUDA:
      return Device(DeviceType::GPU, 0);
    case at::kXLA:
      return *xla_device;
    default:
      XLA_ERROR() << "Device type " << DeviceTypeName(at_device_type, false)
                  << " not supported";
  }
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
