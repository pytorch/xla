#include "cpp_test_util.h"
#include "tensor_impl.h"

#include <string>

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_xla {
namespace cpp_test {

at::Tensor ToTensor(const at::Tensor& tensor) {
  return tensor.is_variable() ? torch::autograd::as_variable_ref(tensor).data()
                              : tensor;
}

at::Tensor ToTensor(XLATensor& xla_tensor) {
  return ToTensor(xla_tensor.ToTensor());
}

at::Tensor ToCpuTensor(const at::Tensor& t) {
  at::Tensor tensor = ToTensor(t);
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

void AllClose(at::Tensor tensor, at::Tensor xla_tensor, double rtol,
              double atol) {
  EXPECT_TRUE(
      ToCpuTensor(xla_tensor).allclose(ToCpuTensor(tensor), rtol, atol));
}

}  // namespace cpp_test
}  // namespace torch_xla
