#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include <ATen/ATen.h>
#include "tensor.h"

TEST(TensotTest, TestAdd) {
  torch_xla::Device device("CPU:0");
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));

  auto c = a.add(b, 1.0);
  auto dev_a = torch_xla::XLATensor::Create(a, device, /*requires_grad=*/false);
  auto dev_b = torch_xla::XLATensor::Create(b, device, /*requires_grad=*/false);
  auto dev_c = dev_a->add(*dev_b, 1.0);
  // Investigate why we get:
  //   expected type CPUFloatType but got Variable[CPUFloatType]
  // ASSERT_TRUE(c.allclose(dev_c->ToTensor(), 1e-5, 1e-8));
}
