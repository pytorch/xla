#pragma once

#include <cmath>
#include <functional>

#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "device.h"
#include "tensor.h"

namespace torch_xla {
namespace cpp_test {

// Converts an XLA ATen tensor to a CPU backend tensor. Extracts it first from
// an autograd variable, if needed. Needed because EqualValues and AllClose
// require CPU tensors on both sides. If the input tensor is already a CPU
// tensor, it will be returned.
at::Tensor ToCpuTensor(const at::Tensor& t);

at::Tensor ToTensor(XLATensor& xla_tensor);

bool EqualValues(at::Tensor a, at::Tensor b);

bool CloseValues(at::Tensor tensor1, at::Tensor tensor2, double rtol = 1e-5,
                 double atol = 1e-8);

static inline void AllClose(at::Tensor tensor, at::Tensor xla_tensor,
                            double rtol = 1e-5, double atol = 1e-8) {
  EXPECT_TRUE(CloseValues(tensor, xla_tensor, rtol, atol));
}

static inline void AllClose(at::Tensor tensor, XLATensor& xla_tensor,
                            double rtol = 1e-5, double atol = 1e-8) {
  EXPECT_TRUE(CloseValues(tensor, ToTensor(xla_tensor), rtol, atol));
}

void ForEachDevice(const std::function<void(const Device&)>& devfn);

}  // namespace cpp_test
}  // namespace torch_xla
