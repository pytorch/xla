#pragma once

#include <cmath>
#include <functional>

#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "device.h"
#include "tensor.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_xla {
namespace cpp_test {

static inline at::Tensor ToTensor(XLATensor& xla_tensor) {
  at::Tensor xtensor = xla_tensor.ToTensor();
  if (xtensor.is_variable()) {
    xtensor = torch::autograd::as_variable_ref(xtensor).data();
  }
  return xtensor;
}

static inline void AllClose(at::Tensor tensor, XLATensor& xla_tensor,
                            double rtol = 1e-5, double atol = 1e-8) {
  EXPECT_TRUE(tensor.allclose(ToTensor(xla_tensor), rtol, atol));
}

void ForEachDevice(const std::function<void(const Device&)>& devfn);

}  // namespace cpp_test
}  // namespace torch_xla
