#include "cpp_test_util.h"
#include "tensor_impl.h"

#include <string>

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_xla {
namespace cpp_test {

at::Tensor ToTensor(XLATensor& xla_tensor) {
  at::Tensor xtensor = xla_tensor.ToTensor();
  if (xtensor.is_variable()) {
    xtensor = torch::autograd::as_variable_ref(xtensor).data();
  }
  return xtensor;
}

at::Tensor ToCpuTensor(const at::Tensor& t) {
  auto impl = dynamic_cast<XLATensorImpl*>(t.unsafeGetTensorImpl());
  CHECK(impl);
  return ToTensor(impl->tensor());
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

void AllClose(at::Tensor tensor, at::Tensor xla_tensor, double rtol /* = 1e-5*/,
              double atol /* = 1e-8*/) {
  EXPECT_TRUE(ToCpuTensor(xla_tensor).allclose(tensor));
}

}  // namespace cpp_test
}  // namespace torch_xla
