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

XLATensor& GetXlaTensor(const at::Tensor& tensor) {
  XLATensorImpl* impl =
      dynamic_cast<XLATensorImpl*>(tensor.unsafeGetTensorImpl());
  XLA_CHECK(impl != nullptr);
  return impl->tensor();
}

std::vector<at::Tensor> XlaCreateTensorList(const at::TensorList& tensors,
                                            const std::vector<bool>* writeable) {
  std::vector<XLATensor> xla_tensors;
  for (auto& tensor : tensors) {
    xla_tensors.push_back(GetXlaTensor(tensor));
  }
  return XLATensor::GetTensors(&xla_tensors, writeable);
}

at::Tensor XlaToAtenTensor(const at::Tensor& tensor) {
  return GetXlaTensor(tensor).ToTensor();
}

at::Tensor XlaToAtenMutableTensor(const at::Tensor& tensor) {
  return GetXlaTensor(tensor).ToMutableTensor();
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

at::Tensor CreateXlaTensor(const at::Tensor& tensor, const Device& device) {
  XLATensor xtensor =
      XLATensor::Create(tensor, device, /*requires_grad=*/false);
  return at::Tensor(
      c10::intrusive_ptr<XLATensorImpl>::make(std::move(xtensor)));
}

}  // namespace bridge
}  // namespace torch_xla
