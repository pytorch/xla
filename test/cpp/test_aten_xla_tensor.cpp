#include <gtest/gtest.h>

#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include "aten_xla_bridge.h"
#include "aten_xla_type_instances.h"
#include "cpp_test_util.h"
#include "tensor_impl.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "torch_util.h"
#include "torch_xla_test.h"

namespace torch_xla {
namespace cpp_test {

class AtenXlaTensorTest : public TorchXlaTest {
 protected:
  static void SetUpTestCase() {
    AtenXlaType::RegisterAtenTypes();
    AtenXlaType::SetFullConvPrecision();
  }
};

at::Tensor GetTestTensor(at::IntList sizes) {
  return at::rand(sizes, at::TensorOptions(at::kFloat));
}

void TestBackward(
    const std::vector<at::Tensor>& inputs, const Device& device,
    const std::function<at::Tensor(const std::vector<at::Tensor>&)>& testfn,
    double rtol = 1e-5, double atol = 1e-8,
    const std::vector<bool>& inputs_require_grad = {}) {
  CHECK(inputs_require_grad.empty() ||
        inputs.size() == inputs_require_grad.size());
  std::vector<at::Tensor> input_vars;
  std::vector<at::Tensor> xinput_vars;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto& input = inputs[i];
    if (input.defined()) {
      const bool requires_grad =
          inputs_require_grad.empty() ? true : inputs_require_grad[i];
      input_vars.push_back(
          torch::autograd::make_variable(input, requires_grad));

      at::Tensor xinput = bridge::CreateXlaTensor(CopyTensor(input), device);
      xinput_vars.push_back(
          torch::autograd::make_variable(xinput, requires_grad));
    } else {
      input_vars.emplace_back();
      xinput_vars.emplace_back();
    }
  }

  at::Tensor output = testfn(input_vars);
  at::Tensor xoutput = testfn(xinput_vars);
  output.backward();
  xoutput.backward();
  for (size_t i = 0; i < input_vars.size(); ++i) {
    if (inputs[i].defined() && input_vars[i].requires_grad()) {
      ASSERT_TRUE(xinput_vars[i].grad().defined());
      AllClose(input_vars[i].grad(), xinput_vars[i].grad(), rtol, atol);
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAdd) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::add(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::add(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestAddInPlace) {
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor c = a.add_(b);
    at::Tensor xla_c = xla_a.add_(xla_b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMul) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::mul(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::mul(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMulInPlace) {
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor c = a.mul_(b);
    at::Tensor xla_c = xla_a.mul_(xla_b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestDiv) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::div(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::div(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestDivInPlace) {
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor c = a.div_(b);
    at::Tensor xla_c = xla_a.div_(xla_b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestNe) {
  at::Tensor a = GetTestTensor({2, 3});
  at::Tensor b = GetTestTensor({2, 3});
  at::Tensor c = at::ne(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::ne(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestEq) {
  at::Tensor a = GetTestTensor({2, 3});
  at::Tensor b = a.clone();
  at::Tensor c = at::eq(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::eq(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestGe) {
  at::Tensor a = GetTestTensor({2, 3});
  at::Tensor b = a.clone();
  at::Tensor c = at::ge(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::ge(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestLe) {
  at::Tensor a = GetTestTensor({2, 3});
  at::Tensor b = a.clone();
  at::Tensor c = at::le(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::le(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestGt) {
  at::Tensor a = GetTestTensor({2, 3});
  at::Tensor b = at::add(a.clone(), at::ones_like(a));
  at::Tensor c = at::gt(b, a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::gt(xla_b, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestLt) {
  at::Tensor a = GetTestTensor({2, 3});
  at::Tensor b = at::add(a.clone(), at::ones_like(a));
  at::Tensor c = at::lt(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::lt(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestNeScalar) {
  at::Tensor input = at::ones({2, 3});
  at::Scalar other(float(0));
  at::Tensor result = at::ne(input, other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::ne(xla_input, other);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestEqScalar) {
  at::Tensor input = at::ones({2, 3});
  at::Scalar other(float(1));
  at::Tensor result = at::eq(input, other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::eq(xla_input, other);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestGeScalar) {
  at::Tensor input = at::ones({2, 3});
  at::Scalar other(float(1));
  at::Tensor result = at::ge(input, other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::ge(xla_input, other);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestLeScalar) {
  at::Tensor input = at::ones({2, 3});
  at::Scalar other(float(1));
  at::Tensor result = at::le(input, other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::le(xla_input, other);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestGtScalar) {
  at::Tensor input = at::ones({2, 3});
  at::Scalar other(float(0.5));
  at::Tensor result = at::gt(input, other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::gt(xla_input, other);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestLtScalar) {
  at::Tensor input = at::ones({2, 3});
  at::Scalar other(float(1.5));
  at::Tensor result = at::lt(input, other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::lt(xla_input, other);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestIntegerAdd) {
  std::vector<at::ScalarType> types(
      {at::kByte, at::kChar, at::kShort, at::kInt, at::kLong});

  ForEachDevice([&](const Device& device) {
    for (auto type : types) {
      at::Tensor a = at::randint(0, 63, {2, 2}, at::TensorOptions(type));
      at::Tensor b = at::randint(0, 63, {2, 2}, at::TensorOptions(type));
      at::Tensor c = at::add(b, 1.0);

      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
      at::Tensor xla_c = at::add(xla_b, 1.0);

      EXPECT_TRUE(EqualValues(c, ToCpuTensor(xla_c)));
    }
  });
}

TEST_F(AtenXlaTensorTest, TestMin) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::min(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::min(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMax) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::max(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::max(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestSin) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::sin(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::sin(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestCos) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::cos(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::cos(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestClampMinMax) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::clamp(a, 0.311, 0.409);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::clamp(xla_a, 0.311, 0.409);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestClampMin) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::clamp(a, 0.311, c10::nullopt);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::clamp(xla_a, 0.311, c10::nullopt);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestClampMax) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::clamp(a, c10::nullopt, 0.409);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::clamp(xla_a, c10::nullopt, 0.409);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestNeg) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::neg(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::neg(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestAbs) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::abs(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::abs(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestAddCMul) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor d = at::addcmul(a, b, c, 3.1165);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
    at::Tensor xla_d = at::addcmul(xla_a, xla_b, xla_c, 3.1165);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenXlaTensorTest, TestAddCDiv) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::abs(at::rand({2, 2}, at::TensorOptions(at::kFloat))) + 1.0;
  at::Tensor d = at::addcdiv(a, b, c, 3.1165);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
    at::Tensor xla_d = at::addcdiv(xla_a, xla_b, xla_c, 3.1165);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenXlaTensorTest, TestSize) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    for (int dim = -rank; dim < rank; ++dim) {
      EXPECT_EQ(at::size(input, dim), at::size(xla_input, dim));
    }
  });
}

TEST_F(AtenXlaTensorTest, TestSlice) {
  at::Tensor a = at::rand({32, 24, 16}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::slice(a, 1, 0, 16, 1);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::slice(xla_a, 1, 0, 16, 1);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestRelu) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::relu(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::relu(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestExp) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::exp(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::exp(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestLog) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::log(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::log(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestSqrt) {
  at::Tensor a = at::abs(at::rand({2, 2}, at::TensorOptions(at::kFloat)));
  at::Tensor b = at::sqrt(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::sqrt(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestPow) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::pow(a, 4.09);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::pow(xla_a, 4.09);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestThreshold) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  float threshold = 0.4;
  float value = 20;
  at::Tensor output = at::threshold(input, threshold, value);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::threshold(xla_input, threshold, value);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestAddMatMul) {
  int in_channels = 32;
  int out_channels = 320;
  int labels = 50;
  at::Tensor input =
      at::rand({in_channels, out_channels}, at::TensorOptions(at::kFloat));
  at::Tensor weight =
      at::rand({out_channels, labels}, at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({labels}, at::TensorOptions(at::kFloat));
  // Test beta != 1. through the CPU interop.
  for (double beta : {1., 2.}) {
    at::Tensor output = at::addmm(bias, input, weight, /*beta=*/beta);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_weight = bridge::CreateXlaTensor(weight, device);
      at::Tensor xla_bias = bridge::CreateXlaTensor(bias, device);
      at::Tensor xla_output =
          at::addmm(xla_bias, xla_input, xla_weight, /*beta=*/beta);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestTranspose) {
  at::Tensor input = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::t(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::t(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestView) {
  at::Tensor input = at::rand({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor output = input.view({-1, 320});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = xla_input.view({-1, 320});
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestViewMod) {
  at::Tensor input = at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor one = at::tensor(1.0, at::TensorOptions(at::kFloat));
  at::Tensor output = input.view({-1, 320});
  output.add_(one, 1.0);
  input.add_(one, 1.0);
  ForEachDevice([&](const Device& device) {
    at::Tensor xinput =
        at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
    at::Tensor xla_input = bridge::CreateXlaTensor(xinput, device);
    at::Tensor xla_one = bridge::CreateXlaTensor(one, device);
    at::Tensor xla_output = xla_input.view({-1, 320});
    xla_output.add_(xla_one, 1.0);
    xla_input.add_(xla_one, 1.0);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestViewModComplex) {
  at::Tensor input = at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor one = at::tensor(1.0, at::TensorOptions(at::kFloat));
  at::Tensor output1 = input.view({-1, 320});
  output1.add_(one, 1.0);
  at::Tensor output2 = input.view({-1, 160});
  output2.add_(one, 1.0);
  ForEachDevice([&](const Device& device) {
    at::Tensor xinput =
        at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
    at::Tensor xla_input = bridge::CreateXlaTensor(xinput, device);
    at::Tensor xla_one = bridge::CreateXlaTensor(one, device);
    at::Tensor xla_output1 = xla_input.view({-1, 320});
    xla_output1.add_(xla_one, 1.0);
    at::Tensor xla_output2 = xla_input.view({-1, 160});
    xla_output2.add_(xla_one, 1.0);
    AllClose(output1, xla_output1);
    AllClose(output2, xla_output2);
  });
}

TEST_F(AtenXlaTensorTest, TestViewOfViewMod) {
  at::Tensor input = at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor one = at::tensor(1.0, at::TensorOptions(at::kFloat));
  at::Tensor output1 = input.view({-1, 320});
  output1.add_(one, 1.0);
  at::Tensor output2 = output1.view({-1, 160});
  output2.add_(one, 1.0);
  ForEachDevice([&](const Device& device) {
    at::Tensor xinput =
        at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
    at::Tensor xla_input = bridge::CreateXlaTensor(xinput, device);
    at::Tensor xla_one = bridge::CreateXlaTensor(one, device);
    at::Tensor xla_output1 = xla_input.view({-1, 320});
    xla_output1.add_(xla_one, 1.0);
    at::Tensor xla_output2 = xla_output1.view({-1, 160});
    xla_output2.add_(xla_one, 1.0);
    AllClose(output1, xla_output1);
    AllClose(output2, xla_output2);
  });
}

TEST_F(AtenXlaTensorTest, TestLogSoftmax) {
  at::Tensor input = at::rand({5, 3, 4, 2}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    for (int dim = 0; dim < input.dim(); ++dim) {
      at::Tensor output = at::log_softmax(input, dim);
      at::Tensor xla_output = at::log_softmax(xla_input, dim);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestSoftmax) {
  at::Tensor input = at::rand({10, 8, 24, 16}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    for (int dim = 0; dim < input.dim(); ++dim) {
      at::Tensor output = at::softmax(input, dim);
      at::Tensor xla_output = at::softmax(xla_input, dim);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestMaxPool2D) {
  at::Tensor input = at::rand({1, 64, 112, 112}, at::TensorOptions(at::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          at::Tensor output = at::max_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output =
                at::max_pool2d(xla_input,
                               /*kernel_size=*/{kernel_size, kernel_size},
                               /*stride=*/{stride, stride},
                               /*padding=*/{padding, padding},
                               /*dilation=*/{dilation, dilation},
                               /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool2DNonSquare) {
  at::Tensor input = at::rand({1, 64, 112, 112}, at::TensorOptions(at::kFloat));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          at::Tensor output = at::max_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size + 1},
              /*stride=*/{stride, stride + 1},
              /*padding=*/{padding, padding + 1},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output =
                at::max_pool2d(xla_input,
                               /*kernel_size=*/{kernel_size, kernel_size + 1},
                               /*stride=*/{stride, stride + 1},
                               /*padding=*/{padding, padding + 1},
                               /*dilation=*/{dilation, dilation},
                               /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAvgPool2D) {
  at::Tensor input = at::rand({4, 1, 28, 28}, at::TensorOptions(at::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          at::Tensor output = at::avg_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output =
                at::avg_pool2d(xla_input,
                               /*kernel_size=*/{kernel_size, kernel_size},
                               /*stride=*/{stride, stride},
                               /*padding=*/{padding, padding},
                               /*ceil_mode=*/ceil_mode,
                               /*count_include_pad=*/count_include_pad);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAvgPool2DNonSquare) {
  at::Tensor input = at::rand({4, 1, 28, 28}, at::TensorOptions(at::kFloat));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          at::Tensor output = at::avg_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size + 1},
              /*stride=*/{stride, stride + 1},
              /*padding=*/{padding, padding + 1}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output =
                at::avg_pool2d(xla_input,
                               /*kernel_size=*/{kernel_size, kernel_size + 1},
                               /*stride=*/{stride, stride + 1},
                               /*padding=*/{padding, padding + 1},
                               /*ceil_mode=*/ceil_mode,
                               /*count_include_pad=*/count_include_pad);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAdaptiveAvgPool2D) {
  at::Tensor input = at::rand({4, 1, 28, 28}, at::TensorOptions(at::kFloat));
  for (int64_t output_size : {7, 8}) {
    at::Tensor output =
        at::adaptive_avg_pool2d(input, {output_size, output_size});
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output =
          at::adaptive_avg_pool2d(xla_input, {output_size, output_size});
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestConv2D) {
  int in_channels = 3;
  int out_channels = 7;
  int kernel_size = 5;
  at::Tensor input =
      at::rand({4, in_channels, 28, 28}, at::TensorOptions(at::kFloat));
  at::Tensor weight =
      at::rand({out_channels, in_channels, kernel_size, kernel_size},
               at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({out_channels}, at::TensorOptions(at::kFloat));
  at::Tensor bias_undef;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          at::Tensor output =
              at::conv2d(input, weight, with_bias ? bias : bias_undef,
                         /*stride=*/{stride, stride},
                         /*padding=*/{padding, padding},
                         /*dilation=*/{dilation, dilation});
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_weight = bridge::CreateXlaTensor(weight, device);
            at::Tensor xla_bias = bridge::CreateXlaTensor(bias, device);
            at::Tensor xla_output = at::conv2d(
                xla_input, xla_weight, with_bias ? xla_bias : bias_undef,
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*dilation=*/{dilation, dilation});
            AllClose(output, xla_output);
          });
        }
      };
    }
  }
}

TEST_F(AtenXlaTensorTest, TestConv2DNonSquare) {
  int in_channels = 3;
  int out_channels = 7;
  int kernel_size = 5;
  at::Tensor input =
      at::rand({4, in_channels, 28, 28}, at::TensorOptions(at::kFloat));
  at::Tensor weight =
      at::rand({out_channels, in_channels, kernel_size, kernel_size},
               at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({out_channels}, at::TensorOptions(at::kFloat));
  at::Tensor bias_undef;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          at::Tensor output =
              at::conv2d(input, weight, with_bias ? bias : bias_undef,
                         /*stride=*/{stride, stride + 1},
                         /*padding=*/{padding, padding + 1},
                         /*dilation=*/{dilation, dilation});
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_weight = bridge::CreateXlaTensor(weight, device);
            at::Tensor xla_bias = bridge::CreateXlaTensor(bias, device);
            at::Tensor xla_output = at::conv2d(
                xla_input, xla_weight, with_bias ? xla_bias : bias_undef,
                /*stride=*/{stride, stride + 1},
                /*padding=*/{padding, padding + 1},
                /*dilation=*/{dilation, dilation});
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestNllLoss) {
  int batch = 3;
  int classes = 5;
  at::Tensor input = GetTestTensor({batch, classes});
  at::Tensor target =
      at::empty({batch}, at::TensorOptions(at::kLong)).random_(0, classes);
  at::Tensor undef_weight;
  for (Reduction::Reduction reduction : {Reduction::Mean, Reduction::Sum}) {
    at::Tensor output =
        at::nll_loss(/*self=*/input, /*target=*/target, /*weight=*/undef_weight,
                     /*reduction=*/reduction);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_target = bridge::CreateXlaTensor(target, device);
      at::Tensor xla_output = at::nll_loss(
          /*self=*/xla_input, /*target=*/xla_target, /*weight=*/undef_weight,
          /*reduction=*/reduction);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestBatchNorm2D) {
  int num_features = 3;
  at::Tensor input = GetTestTensor({14, num_features, 5, 7});
  at::Tensor weight = GetTestTensor({num_features});
  at::Tensor bias = GetTestTensor({num_features});
  at::Tensor running_mean =
      at::zeros({num_features}, at::TensorOptions(at::kFloat));
  at::Tensor running_var =
      at::ones({num_features}, at::TensorOptions(at::kFloat));
  double momentum = 0.1;
  double eps = 0.5;
  for (bool training : {true, false}) {
    at::Tensor output = at::batch_norm(
        /*input=*/input, /*weight=*/weight, /*bias=*/bias,
        /*running_mean=*/running_mean, /*running_var=*/running_var,
        /*training=*/training, /*momentum=*/momentum, /*eps=*/eps,
        /*cudnn_enabled=*/false);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_weight = bridge::CreateXlaTensor(weight, device);
      at::Tensor xla_bias = bridge::CreateXlaTensor(bias, device);
      at::Tensor xla_running_mean =
          bridge::CreateXlaTensor(running_mean, device);
      at::Tensor xla_running_var = bridge::CreateXlaTensor(running_var, device);
      at::Tensor xla_output = at::batch_norm(
          /*input=*/xla_input, /*weight=*/xla_weight, /*bias=*/xla_bias,
          /*running_mean=*/xla_running_mean, /*running_var=*/xla_running_var,
          /*training=*/training, /*momentum=*/momentum, /*eps=*/eps,
          /*cudnn_enabled=*/false);
      AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestDim) {
  at::Tensor input = GetTestTensor({2, 3});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    EXPECT_EQ(input.dim(), xla_input.dim());
  });
}

TEST_F(AtenXlaTensorTest, TestAvgPool2DBackward) {
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          auto testfn =
              [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
            return at::avg_pool2d(inputs[0],
                                  /*kernel_size=*/{kernel_size, kernel_size},
                                  /*stride=*/{stride, stride},
                                  /*padding=*/{padding, padding},
                                  /*ceil_mode=*/ceil_mode,
                                  /*count_include_pad=*/count_include_pad);
          };

          ForEachDevice([&](const Device& device) {
            TestBackward({GetTestTensor({4, 1, 28, 28})}, device, testfn);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAdaptiveAvgPool2DBackward) {
  for (int64_t output_size : {7, 8}) {
    auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
      return at::adaptive_avg_pool2d(inputs[0], {output_size, output_size});
    };
    ForEachDevice([&](const Device& device) {
      TestBackward({GetTestTensor({4, 1, 28, 28})}, device, testfn);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestConv2DBackward) {
  int in_channels = 3;
  int out_channels = 7;
  int kernel_size = 5;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          auto testfn =
              [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
            return at::conv2d(inputs[0], inputs[1], inputs[2],
                              /*stride=*/{stride, stride},
                              /*padding=*/{padding, padding},
                              /*dilation=*/{dilation, dilation});
          };

          ForEachDevice([&](const Device& device) {
            at::Tensor bias =
                with_bias ? GetTestTensor({out_channels}) : at::Tensor();
            TestBackward({GetTestTensor({4, in_channels, 32, 32}),
                          GetTestTensor({out_channels, in_channels, kernel_size,
                                         kernel_size}),
                          bias},
                         device, testfn);
          });
        }
      };
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool2DBackward) {
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
          return at::max_pool2d(
              inputs[0], /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const Device& device) {
          TestBackward({GetTestTensor({1, 64, 112, 112})}, device, testfn);
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestLogSoftmaxBackward) {
  for (int dim = 0; dim < 4; ++dim) {
    auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
      return at::log_softmax(inputs[0], dim);
    };

    ForEachDevice([&](const Device& device) {
      TestBackward({GetTestTensor({5, 3, 4, 2})}, device, testfn, /*rtol=*/1e-3,
                   /*atol=*/1e-5);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestReluBackward) {
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::relu(inputs[0]);
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({GetTestTensor({2, 1, 4, 6})}, device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestTransposeBackward) {
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::t(inputs[0]);
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({GetTestTensor({2, 3})}, device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestAddMatMulBackward) {
  int in_channels = 32;
  int out_channels = 320;
  int labels = 50;
  // Test beta != 1. through the CPU interop.
  for (double beta : {1., 2.}) {
    auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
      return at::addmm(inputs[0], inputs[1], inputs[2], /*beta=*/beta);
    };
    ForEachDevice([&](const Device& device) {
      TestBackward(
          {GetTestTensor({labels}), GetTestTensor({in_channels, out_channels}),
           GetTestTensor({out_channels, labels})},
          device, testfn);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestNllLossBackward) {
  int batch = 3;
  int classes = 5;
  at::Tensor input = GetTestTensor({batch, classes});
  at::Tensor target =
      at::empty({batch}, at::TensorOptions(at::kLong)).random_(0, classes);
  at::Tensor undef_weight;
  for (Reduction::Reduction reduction : {Reduction::Mean, Reduction::Sum}) {
    auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
      return at::nll_loss(
          /*self=*/inputs[0], /*target=*/inputs[1], /*weight=*/undef_weight,
          /*reduction=*/reduction);
    };
    ForEachDevice([&](const Device& device) {
      TestBackward({input, target}, device, testfn, /*rtol=*/1e-5,
                   /*atol=*/1e-8, /*inputs_require_grad=*/{true, false});
    });
  }
}

TEST_F(AtenXlaTensorTest, TestViewBackward) {
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return inputs[0].view({-1, 320});
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({GetTestTensor({32, 20, 4, 4})}, device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestBatchNorm2DBackward) {
  double momentum = 0.1;
  double eps = 0.5;
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::batch_norm(
        /*input=*/inputs[0], /*weight=*/inputs[1], /*bias=*/inputs[2],
        /*running_mean=*/inputs[3], /*running_var=*/inputs[4],
        /*training=*/true, /*momentum=*/momentum, /*eps=*/eps,
        /*cudnn_enabled=*/false);
  };
  int num_features = 3;
  ForEachDevice([&](const Device& device) {
    at::Tensor input = GetTestTensor({14, num_features, 5, 7});
    at::Tensor weight = GetTestTensor({num_features});
    at::Tensor bias = GetTestTensor({num_features});
    at::Tensor running_mean =
        at::zeros({num_features}, at::TensorOptions(at::kFloat));
    at::Tensor running_var =
        at::ones({num_features}, at::TensorOptions(at::kFloat));
    TestBackward({input, weight, bias, running_mean, running_var}, device,
                 testfn,
                 /*rtol=*/1e-3, /*atol=*/1e-4,
                 /*inputs_require_grad=*/{true, true, true, false, false});
  });
}

}  // namespace cpp_test
}  // namespace torch_xla
