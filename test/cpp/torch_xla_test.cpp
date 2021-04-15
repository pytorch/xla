#include "torch_xla_test.h"

#include <ATen/ATen.h>

#include "lazy_tensor_core/csrc/tensor.h"

namespace torch_lazy_tensors {
namespace cpp_test {

void XlaTest::SetUp() {
  at::manual_seed(42);
  LazyTensor::SetRngSeed(GetCurrentDevice(), 42);
}

void XlaTest::TearDown() {}

}  // namespace cpp_test
}  // namespace torch_lazy_tensors
