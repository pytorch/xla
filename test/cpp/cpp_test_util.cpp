#include "cpp_test_util.h"
#include "tensor_impl.h"

#include <iostream>
#include <string>

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch/csrc/autograd/variable.h"
#include "torch_util.h"

namespace torch_xla {
namespace cpp_test {

at::Tensor ToTensor(XLATensor& xla_tensor) {
  return torch_xla::ToTensor(xla_tensor.ToTensor());
}

at::Tensor ToCpuTensor(const at::Tensor& t) {
  at::Tensor tensor = torch_xla::ToTensor(t);
  XLATensorImpl* impl =
      dynamic_cast<XLATensorImpl*>(tensor.unsafeGetTensorImpl());
  return impl != nullptr ? ToTensor(impl->tensor()) : tensor;
}

bool EqualValues(at::Tensor a, at::Tensor b) {
  at::ScalarType atype = a.scalar_type();
  at::ScalarType btype = b.scalar_type();
  if (atype != btype) {
    a = a.toType(btype);
  }
  return a.equal(b);
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

}  // namespace cpp_test
}  // namespace torch_xla
