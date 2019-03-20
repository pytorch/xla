#include "cpp_test_util.h"

#include <iostream>
#include <string>

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch/csrc/autograd/variable.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace cpp_test {

at::Tensor ToTensor(XLATensor& xla_tensor) {
  return torch_xla::ToTensor(xla_tensor.ToTensor());
}

at::Tensor ToCpuTensor(const at::Tensor& t) {
  at::Tensor tensor = torch_xla::ToTensor(t);
  c10::optional<XLATensor> xtensor = bridge::TryGetXlaTensor(tensor);
  return xtensor ? ToTensor(*xtensor) : tensor;
}

bool EqualValues(at::Tensor tensor1, at::Tensor tensor2) {
  tensor1 = ToCpuTensor(tensor1);
  tensor2 = ToCpuTensor(tensor2);

  at::ScalarType type1 = tensor1.scalar_type();
  at::ScalarType type2 = tensor2.scalar_type();
  if (type1 != type2) {
    tensor1 = tensor1.toType(type2);
  }
  return tensor1.equal(tensor2);
}

void ForEachDevice(const std::function<void(const Device&)>& devfn) {
  std::string default_device =
      xla::ComputationClient::Get()->GetDefaultDevice();
  devfn(Device(default_device));
}

bool CloseValues(at::Tensor tensor1, at::Tensor tensor2, double rtol,
                 double atol) {
  tensor1 = ToCpuTensor(tensor1);
  tensor2 = ToCpuTensor(tensor2);
  bool equal = tensor1.allclose(tensor2, rtol, atol);
  if (!equal) {
    std::cout << tensor1 << "\n-vs-\n" << tensor2 << "\n";
  }
  return equal;
}

void WithAllDevices(
    DeviceType device_type,
    const std::function<void(const std::vector<Device>&)>& devfn) {
  std::vector<Device> devices;
  for (const auto& device_str :
       xla::ComputationClient::Get()->GetAvailableDevices()) {
    Device device(device_str);
    if (device.hw_type == device_type) {
      devices.push_back(device);
    }
  }
  if (!devices.empty()) {
    devfn(devices);
  }
}

std::string GetTensorTextGraph(at::Tensor tensor) {
  XLATensor xtensor = bridge::GetXlaTensor(tensor);
  return ir::DumpUtil::ToText({xtensor.GetIrValue().node.get()});
}

std::string GetTensorDotGraph(at::Tensor tensor) {
  XLATensor xtensor = bridge::GetXlaTensor(tensor);
  return ir::DumpUtil::ToDot({xtensor.GetIrValue().node.get()});
}

}  // namespace cpp_test
}  // namespace torch_xla
