#include <gtest/gtest.h>

#include "test/cpp/cpp_test_util.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/op_by_op_executor.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/ops/scalar.h"
#include "torch_xla/csrc/ops/stack.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace cpp_test {

TEST(OpByOpExecutorTest, TestSimpleAdd) {
  if (UsingPjRt()) {
    GTEST_SKIP();
  }

  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    at::Tensor a = at::rand({4, 16, 3}, at::TensorOptions(at::kFloat));
    at::Tensor b = at::rand({4, 16, 3}, at::TensorOptions(at::kFloat));
    at::Tensor c = a + b;

    torch::lazy::Value v_a = GetTensorIrValue(a, device);
    torch::lazy::Value v_b = GetTensorIrValue(b, device);
    torch::lazy::Value v_c = v_a + v_b;

    auto results_data =
        OpByOpExecutor::Get()->Execute({v_c}, device.toString(), {});
    auto results = Fetch(UnwrapXlaData(results_data));

    AllClose(results.front(), c);
  });
}

TEST(OpByOpExecutorTest, TestStack) {
  if (UsingPjRt()) {
    GTEST_SKIP();
  }

  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    at::Tensor a = at::rand({4, 8, 3}, at::TensorOptions(at::kFloat));
    at::Tensor b = at::rand({4, 8, 3}, at::TensorOptions(at::kFloat));
    at::Tensor c = at::stack({a, b}, 1);

    torch::lazy::Value v_a = GetTensorIrValue(a, device);
    torch::lazy::Value v_b = GetTensorIrValue(b, device);
    torch::lazy::Value v_c = torch::lazy::MakeNode<Stack>(
        std::vector<torch::lazy::Value>({v_a, v_b}), 1);

    auto results_data =
        OpByOpExecutor::Get()->Execute({v_c}, device.toString(), {});
    auto results = Fetch(UnwrapXlaData(results_data));

    AllClose(results.front(), c);
  });
}

TEST(OpByOpExecutorTest, TestAsyncStack) {
  if (UsingPjRt()) {
    GTEST_SKIP();
  }

  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    at::Tensor a = at::rand({4, 8, 3}, at::TensorOptions(at::kFloat));
    at::Tensor b = at::rand({4, 8, 3}, at::TensorOptions(at::kFloat));
    at::Tensor c = at::stack({a, b}, 1);

    torch::lazy::Value v_a = GetTensorIrValue(a, device);
    torch::lazy::Value v_b = GetTensorIrValue(b, device);
    torch::lazy::Value v_c = torch::lazy::MakeNode<Stack>(
        std::vector<torch::lazy::Value>({v_a, v_b}), 1);

    auto async =
        OpByOpExecutor::Get()->ExecuteAsync({v_c}, device.toString(), {});
    async.Wait();
    auto results = Fetch(UnwrapXlaData(async.ConsumeValue()));

    AllClose(results.front(), c);
  });
}

}  // namespace cpp_test
}  // namespace torch_xla
