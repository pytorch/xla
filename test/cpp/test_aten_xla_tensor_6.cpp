#include <gtest/gtest.h>
#include <torch/torch.h>

#include <iostream>

#include "test/cpp/cpp_test_util.h"
#include "test/cpp/torch_xla_test.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ops/dynamic_ir.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/runtime/metrics.h"
#include "torch_xla/csrc/torch_util.h"
#include "xla/permutation_util.h"
#include "xla/util.h"

namespace torch_xla {
namespace cpp_test {
namespace {

class AtenXlaTensorTest : public AtenXlaTensorTestBase {};

}  // namespace

TEST_F(AtenXlaTensorTest, TestClamp) {
  torch::Tensor operand =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor min = torch::zeros_like(operand);
  torch::Tensor max = torch::ones_like(operand);
  torch::Tensor out = torch::clamp(operand, min, max);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_operand = CopyToDevice(operand, device);
    torch::Tensor xla_out = torch::clamp(xla_operand, min, max);
    AllClose(out, xla_out, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

}  // namespace cpp_test
}  // namespace torch_xla
