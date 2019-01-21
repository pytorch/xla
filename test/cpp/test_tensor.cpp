#include <gtest/gtest.h>

#include <vector>

#include <ATen/ATen.h>
#include "cpp_test_util.h"
#include "tensor.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_xla {
namespace cpp_test {

TEST(TensotTest, TestAdd) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  auto c = a.add(b, 1.0);

  ForEachDevice([&](const Device& device) {
    auto dev_a = XLATensor::Create(a, device, /*requires_grad=*/false);
    auto dev_b = XLATensor::Create(b, device, /*requires_grad=*/false);
    auto dev_c = dev_a->add(*dev_b, 1.0);

    AllClose(c, *dev_c);
  });
}

TEST(TensotTest, TestIntegerAdd) {
  std::vector<at::ScalarType> types(
      {at::kByte, at::kChar, at::kShort, at::kInt, at::kLong});

  ForEachDevice([&](const Device& device) {
    for (auto type : types) {
      at::Tensor a = at::randint(0, 63, {2, 2}, at::TensorOptions(type));
      at::Tensor b = at::randint(0, 63, {2, 2}, at::TensorOptions(type));
      auto c = a.add(b, 1.0);

      auto dev_a = XLATensor::Create(a, device, /*requires_grad=*/false);
      auto dev_b = XLATensor::Create(b, device, /*requires_grad=*/false);
      auto dev_c = dev_a->add(*dev_b, 1.0);

      EXPECT_TRUE(EqualValues(c, ToTensor(*dev_c)));
    }
  });
}

}  // namespace cpp_test
}  // namespace torch_xla
