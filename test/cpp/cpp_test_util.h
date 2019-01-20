#pragma once

#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include "tensor.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_xla {
namespace cpp_test {

static inline void AllClose(at::Tensor tensor, XLATensor& xla_tensor,
                            double rtol = 1e-5, double atol = 1e-8) {
  EXPECT_TRUE(tensor.allclose(
      torch::autograd::as_variable_ref(xla_tensor.ToTensor()).data(), rtol,
      atol));
}

}  // namespace cpp_test
}  // namespace torch_xla
