#include <ATen/ATen.h>
#include <ATen/LegacyTHFunctions.h>
#include <ATen/NativeFunctions.h>
#include <gtest/gtest.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

#include <iostream>

#include "cpp_test_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla_test.h"

namespace torch_xla {
namespace cpp_test {

class AtenXlaTensorTest : public AtenXlaTensorTestBase {};

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

TEST_F(AtenXlaTensorTest, TestClone) {
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = xla_a.clone();
    AllClose(a, xla_b);
    xla_a.add_(1.0);
    AllClose(a, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestCastByte) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::_cast_Byte(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::_cast_Byte(xla_a);
    EXPECT_TRUE(EqualValues(b, xla_b));
  });
}

TEST_F(AtenXlaTensorTest, TestCastShort) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::_cast_Short(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::_cast_Short(xla_a);
    EXPECT_TRUE(EqualValues(b, xla_b));
  });
}

TEST_F(AtenXlaTensorTest, TestCastInt) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::_cast_Int(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::_cast_Int(xla_a);
    EXPECT_TRUE(EqualValues(b, xla_b));
  });
}

TEST_F(AtenXlaTensorTest, TestCastLong) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::_cast_Long(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::_cast_Long(xla_a);
    EXPECT_TRUE(EqualValues(b, xla_b));
  });
}

TEST_F(AtenXlaTensorTest, TestCastFloat) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::_cast_Float(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::_cast_Float(xla_a);
    EXPECT_TRUE(EqualValues(b, xla_b));
  });
}

TEST_F(AtenXlaTensorTest, TestRetainType) {
  at::Tensor xla_a =
      at::zeros({2, 2}, at::TensorOptions(at::kByte).device(at::kXLA));
  at::Tensor xla_b =
      at::ones({2, 2}, at::TensorOptions(at::kByte).device(at::kXLA));
  at::Tensor xla_c = xla_a + xla_b;
  EXPECT_EQ(xla_c.scalar_type(), at::ScalarType::Byte);
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

TEST_F(AtenXlaTensorTest, TestAddScalar) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar b(1);
  at::Tensor c = at::add(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_c = at::add(xla_a, b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestAddScalarInPlace) {
  at::Scalar b(1);
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor c = a.add_(b);
    at::Tensor xla_c = xla_a.add_(b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestSub) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::sub(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::sub(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestSubInPlace) {
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor c = a.sub_(b);
    at::Tensor xla_c = xla_a.sub_(xla_b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestSubScalar) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar b(1);
  at::Tensor c = at::sub(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_c = at::sub(xla_a, b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestSubScalarInPlace) {
  at::Scalar b(1);
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor c = a.sub_(b);
    at::Tensor xla_c = xla_a.sub_(b);
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

TEST_F(AtenXlaTensorTest, TestMulScalar) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar b(3);
  at::Tensor c = at::mul(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_c = at::mul(xla_a, b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMulScalarInPlace) {
  at::Scalar b(3);
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor c = a.mul_(b);
    at::Tensor xla_c = xla_a.mul_(b);
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

TEST_F(AtenXlaTensorTest, TestDivScalar) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar b(3);
  at::Tensor c = at::div(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_c = at::div(xla_a, b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestDivScalarInPlace) {
  at::Scalar b(3);
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor c = a.div_(b);
    at::Tensor xla_c = xla_a.div_(b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestRsub) {
  at::Tensor input = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor other = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar alpha(2.5);
  at::Tensor result = at::rsub(input, other, alpha);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_other = bridge::CreateXlaTensor(other, device);
    at::Tensor xla_result = at::rsub(xla_input, xla_other, alpha);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestRsubScalar) {
  at::Tensor input = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar other(1.5);
  at::Scalar alpha(2.5);
  at::Tensor result = at::rsub(input, other, alpha);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::rsub(xla_input, other, alpha);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestNe) {
  at::Tensor a = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::ne(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::ne(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestNeInplace) {
  at::Tensor a = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = a.clone();
  b[0] += 1;
  at::Tensor a_copy = a.clone();
  a.ne_(b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a_copy, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    xla_a.ne_(xla_b);
    AllClose(xla_a, a);
  });
}

TEST_F(AtenXlaTensorTest, TestThNe) {
  at::Tensor a = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::_th_ne(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::_th_ne(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestEq) {
  at::Tensor a = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = a.clone();
  at::Tensor c = at::eq(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::eq(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestEqInplace) {
  at::Tensor a = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = a.clone();
  b[0] += 1;
  at::Tensor a_copy = a.clone();
  a.eq_(b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a_copy, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    xla_a.eq_(xla_b);
    AllClose(xla_a, a);
  });
}

TEST_F(AtenXlaTensorTest, TestThEq) {
  at::Tensor a = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = a.clone();
  at::Tensor c = at::_th_eq(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::_th_eq(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestThEqScalar) {
  at::Tensor a = at::full({}, 1.2, at::TensorOptions(at::kFloat));
  at::Tensor b = at::_th_eq(a, 1.2);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::_th_eq(xla_a, 1.2);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestThEqAutograd) {
  at::Tensor a = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = a.clone();
  at::Tensor c = at::_th_eq(torch::autograd::make_variable(a, false),
                            torch::autograd::make_variable(b, false));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = torch::autograd::make_variable(
        bridge::CreateXlaTensor(a, device), false);
    at::Tensor xla_b = torch::autograd::make_variable(
        bridge::CreateXlaTensor(b, device), false);
    at::Tensor xla_c = at::_th_eq(xla_a, xla_b);
    EXPECT_TRUE(EqualValues(c, xla_c));
  });
}

TEST_F(AtenXlaTensorTest, TestGe) {
  at::Tensor a = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = a.clone();
  at::Tensor c = at::ge(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::ge(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestGeInplace) {
  at::Tensor a = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  at::Tensor a_copy = a.clone();
  a.ge_(b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a_copy, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    xla_a.ge_(xla_b);
    AllClose(xla_a, a);
  });
}

TEST_F(AtenXlaTensorTest, TestLe) {
  at::Tensor a = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = a.clone();
  at::Tensor c = at::le(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::le(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestLeInplace) {
  at::Tensor a = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  at::Tensor a_copy = a.clone();
  a.le_(b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a_copy, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    xla_a.le_(xla_b);
    AllClose(xla_a, a);
  });
}

TEST_F(AtenXlaTensorTest, TestThLe) {
  at::Tensor a = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = a.clone();
  at::Tensor c = at::_th_le(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::_th_le(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestGt) {
  at::Tensor a = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::add(a.clone(), at::ones_like(a));
  at::Tensor c = at::gt(b, a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::gt(xla_b, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestGtInplace) {
  at::Tensor a = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  at::Tensor a_copy = a.clone();
  a.gt_(b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a_copy, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    xla_a.gt_(xla_b);
    AllClose(xla_a, a);
  });
}

TEST_F(AtenXlaTensorTest, TestLt) {
  at::Tensor a = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::add(a.clone(), at::ones_like(a));
  at::Tensor c = at::lt(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::lt(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestLtInplace) {
  at::Tensor a = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  at::Tensor a_copy = a.clone();
  a.lt_(b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a_copy, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    xla_a.lt_(xla_b);
    AllClose(xla_a, a);
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

TEST_F(AtenXlaTensorTest, TestThNeScalar) {
  at::Tensor input = at::ones({2, 3});
  at::Scalar other(float(0));
  at::Tensor result = at::_th_ne(input, other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::_th_ne(xla_input, other);
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

TEST_F(AtenXlaTensorTest, TestGeScalarInplace) {
  at::Tensor input = at::arange(-1., 1.5, 0.5, at::TensorOptions(at::kFloat));
  at::Scalar other(float(0));
  at::Tensor input_copy = input.clone();
  input.ge_(other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input_copy, device);
    xla_input.ge_(other);
    AllClose(xla_input, input);
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

TEST_F(AtenXlaTensorTest, TestLeScalarInplace) {
  at::Tensor input = at::arange(-1., 1.5, 0.5, at::TensorOptions(at::kFloat));
  at::Scalar other(float(0));
  at::Tensor input_copy = input.clone();
  input.le_(other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input_copy, device);
    xla_input.le_(other);
    AllClose(xla_input, input);
  });
}

TEST_F(AtenXlaTensorTest, TestThLeScalar) {
  at::Tensor input = at::ones({2, 3});
  at::Scalar other(float(1));
  at::Tensor result = at::_th_le(input, other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::_th_le(xla_input, other);
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

TEST_F(AtenXlaTensorTest, TestGtScalarInplace) {
  at::Tensor input = at::arange(-1., 1.5, 0.5, at::TensorOptions(at::kFloat));
  at::Scalar other(float(0));
  at::Tensor input_copy = input.clone();
  input.gt_(other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input_copy, device);
    xla_input.gt_(other);
    AllClose(xla_input, input);
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

TEST_F(AtenXlaTensorTest, TestLtScalarInplace) {
  at::Tensor input = at::arange(-1., 1.5, 0.5, at::TensorOptions(at::kFloat));
  at::Scalar other(float(0));
  at::Tensor input_copy = input.clone();
  input.lt_(other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input_copy, device);
    xla_input.lt_(other);
    AllClose(xla_input, input);
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

TEST_F(AtenXlaTensorTest, TestSVD) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (auto n : dims) {
      at::Tensor a = at::rand({m, n}, at::TensorOptions(at::kFloat));
      auto b = at::svd(a, /*some=*/true, /*compute_uv=*/true);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
        auto xla_b = at::svd(xla_a, /*some=*/true, /*compute_uv=*/true);
        // The U and V matrices might have different sign for column vectors, so
        // cannot be compared if not by absolute value.
        AllClose(std::get<0>(b).abs(), std::get<0>(xla_b).abs(), /*rtol=*/1e-3,
                 /*atol=*/1e-4);
        at::Tensor diag = std::get<1>(b);
        at::Tensor xla_diag = std::get<1>(xla_b);
        ASSERT_EQ(diag.sizes(), xla_diag.sizes());
        AllClose(diag, xla_diag, /*rtol=*/1e-3,
                 /*atol=*/1e-4);
        AllClose(std::get<2>(b).abs(), std::get<2>(xla_b).abs(), /*rtol=*/1e-3,
                 /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestQR) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (auto n : dims) {
      at::Tensor a = at::rand({m, n}, at::TensorOptions(at::kFloat));
      auto b = at::qr(a);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
        auto xla_b = at::qr(xla_a);
        AllClose(std::get<0>(b).abs(), std::get<0>(xla_b).abs(), /*rtol=*/1e-3,
                 /*atol=*/1e-4);
        AllClose(std::get<1>(b).abs(), std::get<1>(xla_b).abs(), /*rtol=*/1e-3,
                 /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestSymEig) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (bool eigenvectors : {true, false}) {
      for (bool upper : {true, false}) {
        at::Tensor a = at::rand({m, m}, at::TensorOptions(at::kFloat));
        at::Tensor sym_a = a.mm(a.t());
        auto b = at::symeig(sym_a, eigenvectors, upper);
        ForEachDevice([&](const Device& device) {
          at::Tensor xla_a = bridge::CreateXlaTensor(sym_a, device);
          auto xla_b = at::symeig(xla_a, eigenvectors, upper);
          AllClose(std::get<0>(b), std::get<0>(xla_b), /*rtol=*/3e-2,
                   /*atol=*/1e-2);
          AllClose(std::get<1>(b).abs(), std::get<1>(xla_b).abs(),
                   /*rtol=*/3e-2,
                   /*atol=*/1e-2);
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestCholesky) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (bool upper : {true, false}) {
      at::Tensor a = at::rand({3, m, m}, at::TensorOptions(at::kFloat));
      at::Tensor pd_a = at::matmul(a, at::transpose(a, 1, 2)) +
                        at::eye(m, at::TensorOptions(at::kFloat));
      auto b = at::cholesky(pd_a, upper);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_a = bridge::CreateXlaTensor(pd_a, device);
        auto xla_b = at::cholesky(xla_a, upper);
        AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestTriangularSolve) {
  static const int dims[] = {4, 7};
  for (bool batched_a : {true, false}) {
    for (bool batched_b : {true, false}) {
      for (auto m : dims) {
        for (auto n : dims) {
          for (bool upper : {true, false}) {
            for (bool transpose : {true, false}) {
              for (bool unitriangular : {true, false}) {
                at::Tensor a = at::randn({m, m}, at::TensorOptions(at::kFloat));
                at::Tensor b = at::randn({m, n}, at::TensorOptions(at::kFloat));
                a = batched_a ? a.expand({3, m, m}).clone() : a;
                b = batched_b ? b.expand({3, m, n}).clone() : b;
                auto result = at::triangular_solve(
                    b, a, /*upper=*/upper, /*transpose=*/transpose,
                    /*unitriangular=*/unitriangular);
                ForEachDevice([&](const Device& device) {
                  at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
                  at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
                  auto xla_result = at::triangular_solve(
                      xla_b, xla_a, /*upper=*/upper, /*transpose=*/transpose,
                      /*unitriangular=*/unitriangular);
                  AllClose(std::get<0>(result), std::get<0>(xla_result),
                           /*rtol=*/1e-3, /*atol=*/1e-4);
                  AllClose(std::get<1>(result), std::get<1>(xla_result),
                           /*rtol=*/1e-3, /*atol=*/1e-4);
                });
              }
            }
          }
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestKthValue) {
  at::Tensor a = at::rand({4, 5, 3}, at::TensorOptions(at::kFloat));
  for (int k = 1; k <= 3; ++k) {
    int rank = a.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      for (bool keepdim : {false, true}) {
        auto b = at::kthvalue(a, k, dim, keepdim);
        ForEachDevice([&](const Device& device) {
          at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
          auto xla_b = at::kthvalue(xla_a, k, dim, keepdim);
          AllClose(std::get<0>(b), std::get<0>(xla_b));
          AllClose(std::get<1>(b), std::get<1>(xla_b));
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestTopK) {
  at::Tensor a = at::rand({4, 5, 3}, at::TensorOptions(at::kFloat));
  for (int k = 1; k <= 3; ++k) {
    int rank = a.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      for (bool largest : {false, true}) {
        auto b = at::topk(a, k, dim, largest, /*sorted=*/true);
        ForEachDevice([&](const Device& device) {
          at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
          auto xla_b = at::topk(xla_a, k, dim, largest, /*sorted=*/true);
          AllClose(std::get<0>(b), std::get<0>(xla_b));
          AllClose(std::get<1>(b), std::get<1>(xla_b));
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestSort) {
  at::Tensor a = at::rand({4, 5, 3}, at::TensorOptions(at::kFloat));
  for (int k = 1; k <= 3; ++k) {
    for (int dim = 0; dim < 3; ++dim) {
      for (bool descending : {false, true}) {
        auto b = at::sort(a, dim, descending);
        ForEachDevice([&](const Device& device) {
          at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
          auto xla_b = at::sort(xla_a, dim, descending);
          AllClose(std::get<0>(b), std::get<0>(xla_b));
          AllClose(std::get<1>(b), std::get<1>(xla_b));
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestArgSort) {
  at::Tensor a = at::rand({4, 5, 3}, at::TensorOptions(at::kFloat));
  for (int k = 1; k <= 3; ++k) {
    for (int dim = 0; dim < 3; ++dim) {
      for (bool descending : {false, true}) {
        at::Tensor b = at::argsort(a, dim, descending);
        ForEachDevice([&](const Device& device) {
          at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
          at::Tensor xla_b = at::argsort(xla_a, dim, descending);
          AllClose(b, xla_b);
        });
      }
    }
  }
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

TEST_F(AtenXlaTensorTest, TestUnaryMin) {
  at::Tensor input = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::min(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::min(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestUnaryMax) {
  at::Tensor input = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::max(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::max(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestAll) {
  at::Tensor a = at::randint(0, 5, {2, 3, 4}, at::TensorOptions(at::kByte));
  at::Tensor b = at::all(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::all(xla_a);
    EqualValues(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestAllDim) {
  at::Tensor a = at::randint(0, 5, {2, 3, 4}, at::TensorOptions(at::kByte));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor b = at::all(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::all(xla_a, dim, /*keepdim=*/false);
      EqualValues(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestAllDimKeep) {
  at::Tensor a = at::randint(0, 5, {2, 3, 4}, at::TensorOptions(at::kByte));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor b = at::all(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::all(xla_a, dim, /*keepdim=*/true);
      EqualValues(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestAny) {
  at::Tensor a = at::randint(0, 5, {2, 3, 4}, at::TensorOptions(at::kByte));
  at::Tensor b = at::any(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::any(xla_a);
    EqualValues(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestAnyDim) {
  at::Tensor a = at::randint(0, 5, {2, 3, 4}, at::TensorOptions(at::kByte));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor b = at::any(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::any(xla_a, dim, /*keepdim=*/false);
      EqualValues(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestAnyDimKeep) {
  at::Tensor a = at::randint(0, 5, {2, 3, 4}, at::TensorOptions(at::kByte));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor b = at::any(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::any(xla_a, dim, /*keepdim=*/true);
      EqualValues(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMean) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::mean(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::mean(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestMeanInDim) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor b = at::mean(a, {dim});
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::mean(xla_a, {dim});
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMeanInDims) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    at::Tensor b = at::mean(a, dims);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::mean(xla_a, dims);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestSum) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::sum(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::sum(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestSumU8) {
  at::Tensor a = at::ones({256}, at::TensorOptions(at::kByte));
  at::Tensor b = at::sum(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::sum(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestSumInDim) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor b = at::sum(a, {dim});
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::sum(xla_a, {dim});
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestSumInDims) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    at::Tensor b = at::sum(a, dims);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::sum(xla_a, dims);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestSumInDimsKeep) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    at::Tensor b = at::sum(a, dims, /*keepdim=*/true);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::sum(xla_a, dims, /*keepdim=*/true);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMaxInDim) {
  at::Tensor input = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    for (bool keepdim : {false, true}) {
      auto values_indices = at::max(input, dim, /*keepdim=*/keepdim);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
        auto xla_values_indices = at::max(xla_input, dim, /*keepdim=*/keepdim);
        AllClose(std::get<0>(values_indices), std::get<0>(xla_values_indices));
        EXPECT_TRUE(EqualValues(std::get<1>(values_indices),
                                std::get<1>(xla_values_indices)));
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMinInDim) {
  at::Tensor input = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    for (bool keepdim : {false, true}) {
      auto values_indices = at::min(input, dim, /*keepdim=*/keepdim);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
        auto xla_values_indices = at::min(xla_input, dim, /*keepdim=*/keepdim);
        AllClose(std::get<0>(values_indices), std::get<0>(xla_values_indices));
        EXPECT_TRUE(EqualValues(std::get<1>(values_indices),
                                std::get<1>(xla_values_indices)));
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestNorm) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::norm(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::norm(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestNormInDim) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  for (int dim : {1, -2}) {
    at::Tensor b = at::norm(a, 2, {dim}, /*keepdim=*/false);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::norm(xla_a, 2, {dim}, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestNormInDims) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    at::Tensor b = at::norm(a, 2, dims, /*keepdim=*/false);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::norm(xla_a, 2, dims, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestNormInDimsKeep) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    at::Tensor b = at::norm(a, 2, dims, /*keepdim=*/true);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::norm(xla_a, 2, dims, /*keepdim=*/true);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestNormGeneral) {
  at::Tensor a = at::randn({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::norm(a, 3.5);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::norm(xla_a, 3.5);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestNormNuclear) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::norm(a, 1);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::norm(xla_a, 1);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestFrobeniusNorm) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::frobenius_norm(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::frobenius_norm(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestFrobeniusNormInDim) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  for (int dim : {1, -2}) {
    at::Tensor b = at::frobenius_norm(a, {dim}, /*keepdim=*/false);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::frobenius_norm(xla_a, {dim}, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestFrobeniusNormInDims) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    at::Tensor b = at::frobenius_norm(a, dims, /*keepdim=*/false);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::frobenius_norm(xla_a, dims, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestGroupNorm) {
  int num_channels = 6;
  at::Tensor input =
      at::rand({20, num_channels, 10, 10}, at::TensorOptions(at::kFloat));
  at::Tensor weight = at::rand({num_channels}, at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({num_channels}, at::TensorOptions(at::kFloat));
  double eps = 1e-05;
  for (int num_groups : {3, 6, 1}) {
    at::Tensor output = at::group_norm(input, num_groups, weight, bias, eps,
                                       /*cudnn_enabled=*/false);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_weight = bridge::CreateXlaTensor(weight, device);
      at::Tensor xla_bias = bridge::CreateXlaTensor(bias, device);
      at::Tensor xla_output =
          at::group_norm(xla_input, num_groups, xla_weight, xla_bias, eps,
                         /*cudnn_enabled=*/false);
      AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestInstanceNorm) {
  int batch = 5;
  int num_channels = 20;
  at::Tensor input =
      at::rand({batch, num_channels, 10, 10}, at::TensorOptions(at::kFloat));
  at::Tensor weight = at::rand({num_channels}, at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({num_channels}, at::TensorOptions(at::kFloat));
  at::Tensor running_mean =
      at::zeros({num_channels}, at::TensorOptions(at::kFloat));
  at::Tensor running_var =
      at::ones({num_channels}, at::TensorOptions(at::kFloat));
  double momentum = 0.1;
  double eps = 1e-05;
  at::Tensor output = at::instance_norm(
      input, weight, bias, running_mean, running_var,
      /*use_input_stats=*/true, momentum, eps, /*cudnn_enabled=*/false);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_weight = bridge::CreateXlaTensor(weight, device);
    at::Tensor xla_bias = bridge::CreateXlaTensor(bias, device);
    at::Tensor xla_running_mean = bridge::CreateXlaTensor(running_mean, device);
    at::Tensor xla_running_var = bridge::CreateXlaTensor(running_var, device);
    at::Tensor xla_output = at::instance_norm(
        xla_input, xla_weight, xla_bias, xla_running_mean, xla_running_var,
        /*use_input_stats=*/true, momentum, eps, /*cudnn_enabled=*/false);
    AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestLayerNorm) {
  int num_channels = 5;
  std::vector<int64_t> normalized_shape = {10, 10};
  at::Tensor input =
      at::rand({20, num_channels, 10, 10}, at::TensorOptions(at::kFloat));
  at::Tensor weight = at::rand(normalized_shape, at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand(normalized_shape, at::TensorOptions(at::kFloat));
  double eps = 1e-05;
  at::Tensor output = at::layer_norm(input, normalized_shape, weight, bias, eps,
                                     /*cudnn_enabled=*/false);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_weight = bridge::CreateXlaTensor(weight, device);
    at::Tensor xla_bias = bridge::CreateXlaTensor(bias, device);
    at::Tensor xla_output =
        at::layer_norm(xla_input, normalized_shape, xla_weight, xla_bias, eps,
                       /*cudnn_enabled=*/false);
    AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestNuclearNorm) {
  at::Tensor a = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::nuclear_norm(a);
  for (bool keepdim : {false, true}) {
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::nuclear_norm(xla_a);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestPairwiseDistance) {
  at::Tensor x1 = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor x2 = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  double eps = 1e-6;
  for (bool keepdim : {false, true}) {
    for (double p : {1, 2, 3, 4}) {
      ForEachDevice([&](const Device& device) {
        at::Tensor output = at::pairwise_distance(x1, x2, p, eps, keepdim);
        at::Tensor xla_x1 = bridge::CreateXlaTensor(x1, device);
        at::Tensor xla_x2 = bridge::CreateXlaTensor(x2, device);
        at::Tensor xla_output =
            at::pairwise_distance(xla_x1, xla_x2, p, eps, keepdim);
        AllClose(output, xla_output, /*rtol=*/1e-5, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestCosineSimilarity) {
  at::Tensor x1 = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor x2 = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  double eps = 1e-8;
  int rank = x1.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const Device& device) {
      at::Tensor output = at::cosine_similarity(x1, x2, dim, eps);
      at::Tensor xla_x1 = bridge::CreateXlaTensor(x1, device);
      at::Tensor xla_x2 = bridge::CreateXlaTensor(x2, device);
      at::Tensor xla_output = at::cosine_similarity(xla_x1, xla_x2, dim, eps);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCosineEmbeddingLoss) {
  at::Tensor input1 = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor input2 = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor target = at::rand({4}, at::TensorOptions(at::kFloat));
  for (Reduction::Reduction reduction : {Reduction::Mean, Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const Device& device) {
        at::Tensor output = at::cosine_embedding_loss(input1, input2, target,
                                                      margin, reduction);
        at::Tensor xla_input1 = bridge::CreateXlaTensor(input1, device);
        at::Tensor xla_input2 = bridge::CreateXlaTensor(input2, device);
        at::Tensor xla_target = bridge::CreateXlaTensor(target, device);
        at::Tensor xla_output = at::cosine_embedding_loss(
            xla_input1, xla_input2, xla_target, margin, reduction);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestHingeEmbeddingLoss) {
  at::Tensor input = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor target = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  for (Reduction::Reduction reduction : {Reduction::Mean, Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const Device& device) {
        at::Tensor output =
            at::hinge_embedding_loss(input, target, margin, reduction);
        at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
        at::Tensor xla_target = bridge::CreateXlaTensor(target, device);
        at::Tensor xla_output =
            at::hinge_embedding_loss(xla_input, xla_target, margin, reduction);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestTripletMarginLoss) {
  at::Tensor anchor = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor positive =
      at::abs(at::rand({4, 3}, at::TensorOptions(at::kFloat)));
  at::Tensor negative =
      at::neg(at::abs(at::rand({4, 3}, at::TensorOptions(at::kFloat))));
  double eps = 1e-6;
  for (double margin : {0., 0.2}) {
    for (double p : {1, 2, 3, 4}) {
      for (bool swap : {false, true}) {
        for (Reduction::Reduction reduction :
             {Reduction::Mean, Reduction::Sum}) {
          ForEachDevice([&](const Device& device) {
            at::Tensor output = at::triplet_margin_loss(
                anchor, positive, negative, margin, p, eps, swap, reduction);
            at::Tensor xla_anchor = bridge::CreateXlaTensor(anchor, device);
            at::Tensor xla_positive = bridge::CreateXlaTensor(positive, device);
            at::Tensor xla_negative = bridge::CreateXlaTensor(negative, device);
            at::Tensor xla_output =
                at::triplet_margin_loss(xla_anchor, xla_positive, xla_negative,
                                        margin, p, eps, swap, reduction);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMarginRankingLoss) {
  at::Tensor input1 = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor input2 = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor target = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  for (Reduction::Reduction reduction : {Reduction::Mean, Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const Device& device) {
        at::Tensor output =
            at::margin_ranking_loss(input1, input2, target, margin, reduction);
        at::Tensor xla_input1 = bridge::CreateXlaTensor(input1, device);
        at::Tensor xla_input2 = bridge::CreateXlaTensor(input2, device);
        at::Tensor xla_target = bridge::CreateXlaTensor(target, device);
        at::Tensor xla_output = at::margin_ranking_loss(
            xla_input1, xla_input2, xla_target, margin, reduction);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestBCEWithLogits) {
  int batch = 10;
  int classes = 5;
  at::Tensor input = at::rand({batch, classes}, at::TensorOptions(at::kFloat));
  at::Tensor target = at::rand({batch, classes}, at::TensorOptions(at::kFloat));
  at::Tensor weight = at::rand({classes}, at::TensorOptions(at::kFloat));
  at::Tensor pos_weight = at::rand({classes}, at::TensorOptions(at::kFloat));
  at::Tensor undef;
  for (Reduction::Reduction reduction : {Reduction::Mean, Reduction::Sum}) {
    for (bool undef_weight : {false, true}) {
      for (bool undef_pos_weight : {false, true}) {
        ForEachDevice([&](const Device& device) {
          at::Tensor output = at::binary_cross_entropy_with_logits(
              input, target, undef_weight ? undef : weight,
              undef_pos_weight ? undef : pos_weight, reduction);
          at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
          at::Tensor xla_target = bridge::CreateXlaTensor(target, device);
          at::Tensor xla_weight =
              undef_weight ? undef : bridge::CreateXlaTensor(weight, device);
          at::Tensor xla_pos_weight =
              undef_pos_weight ? undef
                               : bridge::CreateXlaTensor(pos_weight, device);
          at::Tensor xla_output = at::binary_cross_entropy_with_logits(
              xla_input, xla_target, xla_weight, xla_pos_weight, reduction);
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestKlDiv) {
  at::Tensor input = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor target = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  for (Reduction::Reduction reduction : {Reduction::Mean, Reduction::Sum}) {
    ForEachDevice([&](const Device& device) {
      at::Tensor output = at::kl_div(input, target, reduction);
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_target = bridge::CreateXlaTensor(target, device);
      at::Tensor xla_output = at::kl_div(xla_input, xla_target, reduction);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestProd) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::prod(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::prod(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestProdInDim) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor b = at::prod(a, dim);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::prod(xla_a, dim);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestProdInDimKeep) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor b = at::prod(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::prod(xla_a, dim, /*keepdim=*/true);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCumSum) {
  at::Tensor input = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor result = at::cumsum(input, dim);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_result = at::cumsum(xla_input, dim);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCumSumCast) {
  at::Tensor input = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor result = at::cumsum(input, dim, at::ScalarType::Int);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_result = at::cumsum(xla_input, dim, at::ScalarType::Int);
      EXPECT_TRUE(EqualValues(result, xla_result));
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCumProd) {
  at::Tensor input = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor result = at::cumprod(input, dim);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_result = at::cumprod(xla_input, dim);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCumProdCast) {
  at::Tensor input =
      at::mul(at::rand({4, 3, 4}, at::TensorOptions(at::kFloat)), 10);
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor result = at::cumprod(input, dim, at::ScalarType::Int);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_result = at::cumprod(xla_input, dim, at::ScalarType::Int);
      EXPECT_TRUE(EqualValues(result, xla_result));
    });
  }
}

TEST_F(AtenXlaTensorTest, TestArgMin) {
  at::Tensor a = at::rand({4, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::argmin(a, c10::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::argmin(xla_a, c10::nullopt, /*keepdim=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestArgMinDim) {
  at::Tensor a = at::rand({4, 4, 4}, at::TensorOptions(at::kFloat));
  for (int dim : {1, -2}) {
    at::Tensor b = at::argmin(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::argmin(xla_a, dim, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestArgMinDimKeep) {
  at::Tensor a = at::rand({4, 4, 4}, at::TensorOptions(at::kFloat));
  for (int dim : {1, -2}) {
    at::Tensor b = at::argmin(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::argmin(xla_a, dim, /*keepdim=*/true);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestArgMinSameValue) {
  at::Tensor a = at::ones({4, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::argmin(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::argmin(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestArgMinWrapper) {
  at::Tensor a = at::rand({4, 4, 4}, at::TensorOptions(at::kFloat));
  for (int dim : {1, -2}) {
    at::Tensor b = at::argmin(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::argmin(xla_a, dim, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestArgMax) {
  at::Tensor a = at::rand({4, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::argmax(a, c10::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::argmax(xla_a, c10::nullopt, /*keepdim=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestArgMaxDim) {
  at::Tensor a = at::rand({4, 4, 4}, at::TensorOptions(at::kFloat));
  for (int dim : {1, -2}) {
    at::Tensor b = at::argmax(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::argmax(xla_a, dim, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestArgMaxDimKeep) {
  at::Tensor a = at::rand({4, 4, 4}, at::TensorOptions(at::kFloat));
  for (int dim : {1, -2}) {
    at::Tensor b = at::argmax(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::argmax(xla_a, dim, /*keepdim=*/true);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestArgMaxSameValue) {
  at::Tensor a = at::ones({4, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::argmax(a, c10::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::argmax(xla_a, c10::nullopt, /*keepdim=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestArgMaxWrapper) {
  at::Tensor a = at::rand({4, 4, 4}, at::TensorOptions(at::kFloat));
  for (int dim : {1, -2}) {
    at::Tensor b = at::argmax(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = at::argmax(xla_a, dim, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestAsin) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::asin(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::asin(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
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

TEST_F(AtenXlaTensorTest, TestSinh) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::sinh(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::sinh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestAcos) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::acos(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::acos(xla_a);
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

TEST_F(AtenXlaTensorTest, TestCosh) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::cosh(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::cosh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestAtan) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::atan(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::atan(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestAtan2) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::randn({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::atan2(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::atan2(xla_a, xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestTan) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::tan(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::tan(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestTanh) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::tanh(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::tanh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestClampMinMax) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar min_val(0.311);
  at::Scalar max_val(0.409);
  at::Tensor b = at::clamp(a, min_val, max_val);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::clamp(xla_a, min_val, max_val);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestClampMin) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar min_val(0.311);
  at::Tensor b = at::clamp(a, min_val, c10::nullopt);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::clamp(xla_a, min_val, c10::nullopt);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestClampMax) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar max_val(0.409);
  at::Tensor b = at::clamp(a, c10::nullopt, max_val);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::clamp(xla_a, c10::nullopt, max_val);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestClampMinExplicit) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar min_val(0.311);
  at::Tensor b = at::clamp_min(a, min_val);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::clamp_min(xla_a, min_val);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestClampMaxExplicit) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar max_val(0.409);
  at::Tensor b = at::clamp_max(a, max_val);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::clamp_max(xla_a, max_val);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestClampMinExplicitInPlace) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar min_val(0.311);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor b = at::clamp_min_(a, min_val);
    at::Tensor xla_b = at::clamp_min_(xla_a, min_val);
    AllClose(a, xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestClampMaxExplicitInPlace) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar max_val(0.409);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor b = at::clamp_max_(a, max_val);
    at::Tensor xla_b = at::clamp_max_(xla_a, max_val);
    AllClose(a, xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestCeil) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::ceil(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::ceil(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestFloor) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::floor(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::floor(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestTrunc) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::trunc(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::trunc(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestFrac) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::frac(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::frac(xla_a);
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

TEST_F(AtenXlaTensorTest, TestSign) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::sign(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::sign(xla_a);
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

TEST_F(AtenXlaTensorTest, TestZeros) {
  at::Tensor a = at::zeros({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor xla_a = at::zeros(
      {2, 2},
      at::TensorOptions(at::kFloat).device(bridge::AtenDefaultDevice()));
  AllClose(a, xla_a);
}

TEST_F(AtenXlaTensorTest, TestOnes) {
  at::Tensor a = at::ones({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor xla_a = at::ones(
      {2, 2},
      at::TensorOptions(at::kFloat).device(bridge::AtenDefaultDevice()));
  AllClose(a, xla_a);
}

TEST_F(AtenXlaTensorTest, TestFull) {
  at::Tensor a = at::full({2, 2}, 3.1165, at::TensorOptions(at::kFloat));
  at::Tensor xla_a = at::full(
      {2, 2}, 3.1165,
      at::TensorOptions(at::kFloat).device(bridge::AtenDefaultDevice()));
  AllClose(a, xla_a);
}

TEST_F(AtenXlaTensorTest, TestARange) {
  at::Tensor a = at::arange(0.0, 100.0, 0.5, at::TensorOptions(at::kFloat));
  at::Tensor xla_a = at::arange(
      0.0, 100.0, 0.5,
      at::TensorOptions(at::kFloat).device(bridge::AtenDefaultDevice()));
  AllClose(a, xla_a);
}

TEST_F(AtenXlaTensorTest, TestBartlettWindow) {
  int window_length = 10;
  for (bool periodic : {false, true}) {
    at::Tensor output = at::bartlett_window(window_length, periodic,
                                            at::TensorOptions(at::kFloat));
    at::Tensor xla_output = at::bartlett_window(
        window_length, periodic,
        at::TensorOptions(at::kFloat).device(bridge::AtenDefaultDevice()));
    AllClose(output, xla_output);
  }
}

TEST_F(AtenXlaTensorTest, TestBlackmanWindow) {
  int window_length = 10;
  for (bool periodic : {false, true}) {
    at::Tensor output = at::blackman_window(window_length, periodic,
                                            at::TensorOptions(at::kFloat));
    at::Tensor xla_output = at::blackman_window(
        window_length, periodic,
        at::TensorOptions(at::kFloat).device(bridge::AtenDefaultDevice()));
    AllClose(output, xla_output, /*rtol=*/1e-5, /*atol=*/1e-7);
  }
}

TEST_F(AtenXlaTensorTest, TestHammingWindow) {
  double alpha = 0.54;
  double beta = 0.46;
  int window_length = 10;
  for (bool periodic : {false, true}) {
    at::Tensor output = at::hamming_window(window_length, periodic, alpha, beta,
                                           at::TensorOptions(at::kFloat));
    at::Tensor xla_output = at::hamming_window(
        window_length, periodic, alpha, beta,
        at::TensorOptions(at::kFloat).device(bridge::AtenDefaultDevice()));
    AllClose(output, xla_output);
  }
}

TEST_F(AtenXlaTensorTest, TestLogSigmoid) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::log_sigmoid(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::log_sigmoid(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestSigmoid) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::sigmoid(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::sigmoid(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestMatmul_1x1) {
  at::Tensor a = at::rand({4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({4}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::matmul(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::matmul(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMatmul_2x1) {
  at::Tensor a = at::rand({3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({4}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::matmul(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::matmul(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMatmul_1x2) {
  at::Tensor a = at::rand({4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::matmul(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::matmul(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMatmul_2x2) {
  at::Tensor a = at::rand({2, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::matmul(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::matmul(xla_a, xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(AtenXlaTensorTest, TestMatmulBcast) {
  at::Tensor a = at::rand({4, 2, 3, 2, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 1, 4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::matmul(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::matmul(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestDot) {
  at::Tensor a = at::rand({4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({4}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::dot(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::dot(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestTensorDot) {
  at::Tensor a = at::rand({6, 4, 8}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({4, 7, 8}, at::TensorOptions(at::kFloat));
  std::vector<int64_t> dims_a = {1, 2};
  std::vector<int64_t> dims_b = {0, 2};
  at::Tensor c = at::tensordot(a, b, dims_a, dims_b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::tensordot(xla_a, xla_b, dims_a, dims_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestBatchMatMul) {
  at::Tensor a = at::rand({3, 6, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({3, 4, 5}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::bmm(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::bmm(xla_a, xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(AtenXlaTensorTest, TestChainMatMul) {
  at::Tensor a = at::rand({5, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({4, 6}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::rand({6, 2}, at::TensorOptions(at::kFloat));
  at::Tensor d = at::rand({2, 7}, at::TensorOptions(at::kFloat));
  at::Tensor result = at::chain_matmul({a, b, c, d});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
    at::Tensor xla_d = bridge::CreateXlaTensor(d, device);
    at::Tensor xla_result = at::chain_matmul({xla_a, xla_b, xla_c, xla_d});
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(AtenXlaTensorTest, TestLinear) {
  at::Tensor input = at::rand({2, 4}, at::TensorOptions(at::kFloat));
  at::Tensor weight = at::rand({3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({3});
  at::Tensor result = at::linear(input, weight);
  at::Tensor result_with_bias = at::linear(input, weight, bias);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_weight = bridge::CreateXlaTensor(weight, device);
    at::Tensor xla_bias = bridge::CreateXlaTensor(bias, device);
    at::Tensor xla_result = at::linear(xla_input, xla_weight);
    at::Tensor xla_result_with_bias =
        at::linear(xla_input, xla_weight, xla_bias);
    AllClose(result, xla_result, /*rtol=*/1e-2, /*atol=*/1e-4);
    AllClose(result_with_bias, xla_result_with_bias, /*rtol=*/1e-2,
             /*atol=*/1e-4);
  });
}

TEST_F(AtenXlaTensorTest, TestPinverse) {
  at::Tensor input = at::rand({4, 6}, at::TensorOptions(at::kFloat));
  at::Tensor result = at::pinverse(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::pinverse(xla_input);
    AllClose(result, xla_result, /*rtol=*/1e-4);
  });
}

TEST_F(AtenXlaTensorTest, TestEinsumOuter) {
  at::Tensor a = at::rand({5}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({5}, at::TensorOptions(at::kFloat));
  std::string equation = "i,j->ij";
  at::Tensor c = at::einsum(equation, {a, b});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::einsum(equation, {xla_a, xla_b});
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestEinsumBatchMatMul) {
  at::Tensor a = at::rand({3, 2, 5}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({3, 5, 4}, at::TensorOptions(at::kFloat));
  std::string equation = "bij,bjk->bik";
  at::Tensor c = at::einsum(equation, {a, b});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::einsum(equation, {xla_a, xla_b});
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerBilinear) {
  at::Tensor a = at::rand({3, 5, 4}, at::TensorOptions(at::kFloat));
  at::Tensor l = at::rand({2, 5}, at::TensorOptions(at::kFloat));
  at::Tensor r = at::rand({2, 4}, at::TensorOptions(at::kFloat));
  std::string equation = "bn,anm,bm->ba";
  at::Tensor c = at::einsum(equation, {l, a, r});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_l = bridge::CreateXlaTensor(l, device);
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_r = bridge::CreateXlaTensor(r, device);
    at::Tensor xla_c = at::einsum(equation, {xla_l, xla_a, xla_r});
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerDiagonal) {
  at::Tensor input = at::rand({3, 3}, at::TensorOptions(at::kFloat));
  std::string equation = "ii->i";
  at::Tensor result = at::einsum(equation, {input});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::einsum(equation, {xla_input});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerBatchDiagonal) {
  at::Tensor input = at::rand({4, 3, 3}, at::TensorOptions(at::kFloat));
  std::string equation = "...ii->...i";
  at::Tensor result = at::einsum(equation, {input});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::einsum(equation, {xla_input});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerBatchPermute) {
  at::Tensor input = at::rand({2, 3, 4, 5}, at::TensorOptions(at::kFloat));
  std::string equation = "...ij->...ji";
  at::Tensor result = at::einsum(equation, {input});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::einsum(equation, {xla_input});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerRepeatedAxis) {
  at::Tensor x = at::rand({2, 3, 3}, at::TensorOptions(at::kFloat));
  at::Tensor y = at::rand({4}, at::TensorOptions(at::kFloat));
  std::string equation = "ijj,k->ik";
  at::Tensor result = at::einsum(equation, {x, y});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_x = bridge::CreateXlaTensor(x, device);
    at::Tensor xla_y = bridge::CreateXlaTensor(y, device);
    at::Tensor xla_result = at::einsum(equation, {xla_x, xla_y});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestBilinear) {
  int batch_size = 16;
  int in1_features = 4;
  int in2_features = 6;
  int out_features = 8;
  at::Tensor input1 =
      at::rand({batch_size, in1_features}, at::TensorOptions(at::kFloat));
  at::Tensor input2 =
      at::rand({batch_size, in2_features}, at::TensorOptions(at::kFloat));
  at::Tensor weight = at::rand({out_features, in1_features, in2_features},
                               at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({out_features}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input1 = bridge::CreateXlaTensor(input1, device);
    at::Tensor xla_input2 = bridge::CreateXlaTensor(input2, device);
    at::Tensor xla_weight = bridge::CreateXlaTensor(weight, device);
    at::Tensor xla_bias = bridge::CreateXlaTensor(bias, device);
    at::Tensor result = at::bilinear(input1, input2, weight, bias);
    at::Tensor xla_result =
        at::bilinear(xla_input1, xla_input2, xla_weight, xla_bias);
    AllClose(result, xla_result);
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

TEST_F(AtenXlaTensorTest, TestSelect) {
  at::Tensor input = at::rand({14, 24, 8}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    for (int dim = -rank; dim < rank; ++dim) {
      at::Tensor output = at::select(input, dim, 4);
      at::Tensor xla_output = at::select(xla_input, dim, 4);
      AllClose(output, xla_output);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestBernoulliScalarProb) {
  at::Tensor input = at::zeros(1000, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::bernoulli(xla_input, 0.1);
    double frac = xla_output.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(AtenXlaTensorTest, TestBernoulliTensorProb) {
  std::vector<float> prob_values(1000, 0.1);
  at::Tensor input = at::tensor(prob_values, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::bernoulli(xla_input);
    double frac = xla_output.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(AtenXlaTensorTest, TestBernoulliScalarProbInPlace) {
  at::Tensor input = at::zeros(1000, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    xla_input.bernoulli_(0.1);
    double frac = xla_input.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(AtenXlaTensorTest, TestBernoulliTensorProbInPlace) {
  at::Tensor input = at::zeros(1000, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor prob = at::scalar_tensor(0.1, at::TensorOptions(at::kFloat));
    xla_input.bernoulli_(bridge::CreateXlaTensor(prob, device));
    double frac = xla_input.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(AtenXlaTensorTest, TestDropout) {
  at::Tensor a = at::rand({17, 21}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor b = at::dropout(xla_a, 0.1, /*train=*/true).cpu();
    double prob =
        static_cast<double>(b.ne(0.0f).sum().item().toDouble()) / a.numel();
    EXPECT_GT(prob, 0.06);
    EXPECT_LT(prob, 0.14);
  });
}

TEST_F(AtenXlaTensorTest, TestDropoutInPlace) {
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::rand({17, 21}, at::TensorOptions(at::kFloat));
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::dropout_(xla_a, 0.1, /*train=*/true);
    double prob =
        static_cast<double>(xla_a.cpu().ne(0.0f).sum().item().toDouble()) /
        a.numel();
    EXPECT_GT(prob, 0.06);
    EXPECT_LT(prob, 0.14);
  });
}

TEST_F(AtenXlaTensorTest, TestRandperm) {
  int n = 5;
  at::Tensor shuffle = at::randperm(n, at::TensorOptions().device(at::kXLA));
  at::Tensor shuffle_cpu = ToCpuTensor(shuffle);
  std::vector<xla::int64> shuffle_data(shuffle_cpu.data<int64_t>(),
                                       shuffle_cpu.data<int64_t>() + n);
  EXPECT_TRUE(xla::IsPermutation(shuffle_data, n));
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

TEST_F(AtenXlaTensorTest, TestStack) {
  at::Tensor a = at::rand({2, 4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::rand({2, 4, 3}, at::TensorOptions(at::kFloat));
  int rank = a.dim() + 1;
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor d = at::stack({a, b, c}, dim);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
      at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
      at::Tensor xla_d = at::stack({xla_a, xla_b, xla_c}, dim);
      AllClose(d, xla_d);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCat) {
  at::Tensor a = at::rand({2, 1, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::rand({2, 3, 3}, at::TensorOptions(at::kFloat));
  int rank = a.dim();
  for (int dim : {1, -2}) {
    at::Tensor d = at::cat({a, b, c}, dim);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
      at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
      at::Tensor xla_d = at::cat({xla_a, xla_b, xla_c}, dim);
      AllClose(d, xla_d);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestUnbind) {
  at::Tensor input = at::rand({4, 3, 7}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    std::vector<at::Tensor> output = at::unbind(input, dim);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      std::vector<at::Tensor> xla_output = at::unbind(xla_input, dim);
      ASSERT_EQ(output.size(), xla_output.size());
      for (size_t i = 0; i < output.size(); ++i) {
        AllClose(output[i], xla_output[i]);
      }
    });
  }
}

TEST_F(AtenXlaTensorTest, TestRepeat) {
  std::vector<std::vector<int64_t>> repeats_list = {{4, 2}, {4, 2, 3}};
  std::vector<std::vector<int64_t>> input_size_list = {{3}, {2, 4}};
  for (const auto& repeats : repeats_list) {
    for (const auto& input_size : input_size_list) {
      at::Tensor input = at::rand(input_size, at::TensorOptions(at::kFloat));
      at::Tensor output = input.repeat(repeats);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
        at::Tensor xla_output = xla_input.repeat(repeats);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestGather) {
  at::Tensor a = at::rand({3, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::empty({3, 3}, at::TensorOptions(at::kLong));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      b[i][j] = (i + j) % 3;
    }
  }
  for (bool sparse_grad : {false, true}) {
    at::Tensor c = at::gather(a, 1, b, sparse_grad);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
      at::Tensor xla_c = at::gather(xla_a, 1, xla_b, sparse_grad);
      AllClose(c, xla_c);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestScatter) {
  at::Tensor a = at::rand({4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::empty({4, 4}, at::TensorOptions(at::kLong));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    at::Tensor d = at::scatter(a, dim, c, b);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
      at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
      at::Tensor xla_d = at::scatter(xla_a, dim, xla_c, xla_b);
      AllClose(d, xla_d);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestScatterBiggerSource) {
  at::Tensor a = at::rand({4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({8, 8}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::empty({4, 4}, at::TensorOptions(at::kLong));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    at::Tensor d = at::scatter(a, dim, c, b);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
      at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
      at::Tensor xla_d = at::scatter(xla_a, dim, xla_c, xla_b);
      AllClose(d, xla_d);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestScatterScalar) {
  at::Tensor a = at::rand({4, 4}, at::TensorOptions(at::kFloat));
  at::Scalar b = 1.0f;
  at::Tensor c = at::empty({4, 4}, at::TensorOptions(at::kLong));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    at::Tensor d = at::scatter(a, dim, c, b);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
      at::Tensor xla_d = at::scatter(xla_a, dim, xla_c, b);
      AllClose(d, xla_d);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestScatterAdd) {
  at::Tensor a = at::rand({4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::empty({4, 4}, at::TensorOptions(at::kLong));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    at::Tensor d = at::scatter_add(a, dim, c, b);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
      at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
      at::Tensor xla_d = at::scatter_add(xla_a, dim, xla_c, xla_b);
      AllClose(d, xla_d);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestScatterAddInPlace) {
  at::Tensor b = at::rand({4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::empty({4, 4}, at::TensorOptions(at::kLong));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    ForEachDevice([&](const Device& device) {
      at::Tensor a = at::rand({4, 4}, at::TensorOptions(at::kFloat));
      at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
      at::Tensor d = a.scatter_add_(dim, c, b);
      at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
      at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
      at::Tensor xla_d = xla_a.scatter_add_(dim, xla_c, xla_b);
      AllClose(d, xla_d);
      AllClose(a, xla_a);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestIndexSelect) {
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor a =
        isFloatingType(scalar_type)
            ? at::rand({3, 4}, at::TensorOptions(scalar_type))
            : at::randint(100, {3, 4}, at::TensorOptions(scalar_type));
    at::Tensor b = at::empty({2}, at::TensorOptions(at::kLong));
    b[0] = 0;
    b[1] = 2;
    at::Tensor c0 = at::index_select(a, 0, b);
    at::Tensor c1 = at::index_select(a, 1, b);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
      at::Tensor xla_c0 = at::index_select(xla_a, 0, xla_b);
      at::Tensor xla_c1 = at::index_select(xla_a, 1, xla_b);
      AllClose(c0, xla_c0);
      AllClose(c1, xla_c1);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestExpand) {
  at::Tensor a = at::rand({3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::native::expand(a, {2, 3, 4}, /*implicit=*/false);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::native::expand(xla_a, {2, 3, 4}, /*implicit=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestExpandBack) {
  at::Tensor a = at::rand({3, 1}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::native::expand(a, {3, 4}, /*implicit=*/false);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::native::expand(xla_a, {3, 4}, /*implicit=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestExpandAs) {
  at::Tensor a = at::rand({3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::native::expand_as(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::native::expand_as(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestEye) {
  int n = 5;
  at::Tensor out = at::eye(n, at::TensorOptions(at::kFloat));
  at::Tensor xla_out = at::eye(
      n, at::TensorOptions(at::kFloat).device(bridge::AtenDefaultDevice()));
  AllClose(out, xla_out);
}

TEST_F(AtenXlaTensorTest, TestEyeWide) {
  int lines = 3;
  int cols = 5;
  at::Tensor out = at::eye(lines, cols, at::TensorOptions(at::kFloat));
  at::Tensor xla_out = at::eye(
      lines, cols,
      at::TensorOptions(at::kFloat).device(bridge::AtenDefaultDevice()));
  AllClose(out, xla_out);
}

TEST_F(AtenXlaTensorTest, TestEyeNarrow) {
  int lines = 5;
  int cols = 3;
  at::Tensor out = at::eye(lines, cols, at::TensorOptions(at::kFloat));
  at::Tensor xla_out = at::eye(
      lines, cols,
      at::TensorOptions(at::kFloat).device(bridge::AtenDefaultDevice()));
  AllClose(out, xla_out);
}

TEST_F(AtenXlaTensorTest, TestBroadcastTensors) {
  at::Tensor a = at::rand({2, 1, 1}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 1}, at::TensorOptions(at::kFloat));
  std::vector<at::Tensor> c = at::broadcast_tensors({a, b});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    std::vector<at::Tensor> xla_c = at::broadcast_tensors({xla_a, xla_b});
    ASSERT_EQ(c.size(), xla_c.size());
    for (size_t i = 0; i < c.size(); ++i) {
      AllClose(c[i], xla_c[i]);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestSCopy) {
  int size = 5;
  at::Tensor source = at::randint(100, {size}, at::TensorOptions(at::kInt));
  at::Tensor destination = at::empty({size}, at::TensorOptions(at::kFloat));
  at::s_copy_(destination, source);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_source = bridge::CreateXlaTensor(source, device);
    at::Tensor xla_destination =
        at::empty({size}, at::TensorOptions(at::kFloat)
                              .device(bridge::XlaDeviceToAtenDevice(device)));
    at::s_copy_(xla_destination, xla_source);
    EXPECT_TRUE(EqualValues(destination, xla_destination));
  });
}

TEST_F(AtenXlaTensorTest, TestSCopyTransfer) {
  int size = 5;
  at::Tensor source = at::randint(100, {size}, at::TensorOptions(at::kInt));
  at::Tensor destination = at::empty({size}, at::TensorOptions(at::kFloat));
  at::s_copy_(destination, source);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_destination =
        at::empty({size}, at::TensorOptions(at::kFloat)
                              .device(bridge::XlaDeviceToAtenDevice(device)));
    at::s_copy_(xla_destination, source);
    EXPECT_TRUE(EqualValues(destination, xla_destination));
  });
}

TEST_F(AtenXlaTensorTest, TestSCopyFrom) {
  int size = 5;
  at::Tensor source = at::randint(100, {size}, at::TensorOptions(at::kInt));
  at::Tensor destination = at::empty({size}, at::TensorOptions(at::kFloat));
  at::s_copy_(destination, source);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_source = bridge::CreateXlaTensor(source, device);
    at::Tensor xla_destination =
        at::empty({size}, at::TensorOptions(at::kFloat)
                              .device(bridge::XlaDeviceToAtenDevice(device)));
    at::_s_copy_from(xla_source, xla_destination);
    EXPECT_TRUE(EqualValues(destination, xla_destination));
  });
}

TEST_F(AtenXlaTensorTest, TestOneIndex) {
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({4, 3, 5, 6, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {4, 3, 5, 6, 7}, at::TensorOptions(scalar_type));
    at::Tensor indices =
        at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
    at::Tensor result = at::index(params, {indices});
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
      at::Tensor xla_indices = bridge::CreateXlaTensor(indices, device);
      at::Tensor xla_result = at::index(xla_params, {xla_indices});
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestOneIndexTransfer) {
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({4, 3, 5, 6, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {4, 3, 5, 6, 7}, at::TensorOptions(scalar_type));
    at::Tensor indices =
        at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
    at::Tensor result = at::index(params, {indices});
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
      at::Tensor xla_result = at::index(xla_params, {indices});
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexHeadNull) {
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({4, 3, 5, 6, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {4, 3, 5, 6, 7}, at::TensorOptions(scalar_type));
    at::Tensor indices_null;
    at::Tensor indices_0 =
        at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
    at::Tensor indices_1 =
        at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
    at::Tensor result = at::index(params, {indices_null, indices_0, indices_1});
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
      at::Tensor xla_indices_0 = bridge::CreateXlaTensor(indices_0, device);
      at::Tensor xla_indices_1 = bridge::CreateXlaTensor(indices_1, device);
      at::Tensor xla_result =
          at::index(xla_params, {indices_null, xla_indices_0, xla_indices_1});
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexMiddleNull) {
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({4, 3, 5, 6, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {4, 3, 5, 6, 7}, at::TensorOptions(scalar_type));
    at::Tensor indices_0 =
        at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
    at::Tensor indices_null;
    at::Tensor indices_1 =
        at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
    at::Tensor result = at::index(params, {indices_0, indices_null, indices_1});
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
      at::Tensor xla_indices_0 = bridge::CreateXlaTensor(indices_0, device);
      at::Tensor xla_indices_1 = bridge::CreateXlaTensor(indices_1, device);
      at::Tensor xla_result =
          at::index(xla_params, {xla_indices_0, indices_null, xla_indices_1});
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexTailNull) {
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({4, 3, 5, 6, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {4, 3, 5, 6, 7}, at::TensorOptions(scalar_type));
    at::Tensor indices_0 =
        at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
    at::Tensor indices_null;
    at::Tensor indices_1 =
        at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
    at::Tensor result = at::index(params, {indices_0, indices_1, indices_null});
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
      at::Tensor xla_indices_0 = bridge::CreateXlaTensor(indices_0, device);
      at::Tensor xla_indices_1 = bridge::CreateXlaTensor(indices_1, device);
      at::Tensor xla_result =
          at::index(xla_params, {xla_indices_0, xla_indices_1, indices_null});
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexMiddleBroadcast) {
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({4, 3, 5, 6, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {4, 3, 5, 6, 7}, at::TensorOptions(scalar_type));
    at::Tensor indices_0 =
        at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
    at::Tensor indices_1 =
        at::randint(-3, 3, {2, 1, 3}, at::TensorOptions(at::kLong));
    at::Tensor result = at::index(params, {indices_0, indices_1});
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
      at::Tensor xla_indices_0 = bridge::CreateXlaTensor(indices_0, device);
      at::Tensor xla_indices_1 = bridge::CreateXlaTensor(indices_1, device);
      at::Tensor xla_result =
          at::index(xla_params, {xla_indices_0, xla_indices_1});
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexTailBroadcast) {
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({4, 3, 5, 6, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {4, 3, 5, 6, 7}, at::TensorOptions(scalar_type));
    at::Tensor indices_0 =
        at::randint(-3, 3, {2, 1, 3}, at::TensorOptions(at::kLong));
    at::Tensor indices_1 =
        at::randint(-3, 3, {2, 1}, at::TensorOptions(at::kLong));
    at::Tensor result = at::index(params, {indices_0, indices_1});
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
      at::Tensor xla_indices_0 = bridge::CreateXlaTensor(indices_0, device);
      at::Tensor xla_indices_1 = bridge::CreateXlaTensor(indices_1, device);
      at::Tensor xla_result =
          at::index(xla_params, {xla_indices_0, xla_indices_1});
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMaskIndex) {
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({2, 2}, at::TensorOptions(scalar_type))
            : at::randint(100, {2, 2}, at::TensorOptions(scalar_type));
    at::Tensor indices =
        at::randint(0, 2, {2, 2}, at::TensorOptions(at::kByte));
    at::Tensor result = at::index(params, {indices});
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
      at::Tensor xla_indices = bridge::CreateXlaTensor(indices, device);
      at::Tensor xla_result = at::index(xla_params, {xla_indices});
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestOneIndexPut) {
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({4, 3, 5, 6, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {4, 3, 5, 6, 7}, at::TensorOptions(scalar_type));
    at::Tensor indices =
        at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
    at::Tensor values =
        isFloatingType(scalar_type)
            ? at::rand({3, 5, 6, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {3, 5, 6, 7}, at::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      at::Tensor result = at::index_put(params, {indices}, values, accumulate);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
        at::Tensor xla_indices = bridge::CreateXlaTensor(indices, device);
        at::Tensor xla_values = bridge::CreateXlaTensor(values, device);
        at::Tensor xla_result =
            at::index_put(xla_params, {xla_indices}, xla_values, accumulate);
        AllClose(result, xla_result);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestOneIndexPutInPlace) {
  at::Tensor indices =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor values = at::ones({3, 5, 6, 7}, at::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      ForEachDevice([&](const Device& device) {
        at::Tensor params =
            isFloatingType(scalar_type)
                ? at::rand({4, 3, 5, 6, 7}, at::TensorOptions(scalar_type))
                : at::randint(100, {4, 3, 5, 6, 7},
                              at::TensorOptions(scalar_type));
        at::Tensor xla_params = bridge::CreateXlaTensor(params.clone(), device);
        at::Tensor result =
            at::index_put_(params, {indices}, values, accumulate);
        at::Tensor xla_indices = bridge::CreateXlaTensor(indices, device);
        at::Tensor xla_values = bridge::CreateXlaTensor(values, device);
        at::Tensor xla_result =
            at::index_put_(xla_params, {xla_indices}, xla_values, accumulate);
        AllClose(result, xla_result);
        AllClose(params, xla_params);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestOneIndexPutTransfer) {
  at::Tensor indices =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({4, 3, 5, 6, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {4, 3, 5, 6, 7}, at::TensorOptions(scalar_type));
    at::Tensor values = at::ones({3, 5, 6, 7}, at::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      at::Tensor result = at::index_put(params, {indices}, values, accumulate);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
        at::Tensor xla_values = bridge::CreateXlaTensor(values, device);
        at::Tensor xla_result =
            at::index_put(xla_params, {indices}, xla_values, accumulate);
        AllClose(result, xla_result);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexPut) {
  at::Tensor indices_0 =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  at::Tensor indices_1 =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({4, 3, 5, 6, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {4, 3, 5, 6, 7}, at::TensorOptions(scalar_type));
    at::Tensor values = at::ones({5, 6, 7}, at::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      at::Tensor result =
          at::index_put(params, {indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
        at::Tensor xla_indices_0 = bridge::CreateXlaTensor(indices_0, device);
        at::Tensor xla_indices_1 = bridge::CreateXlaTensor(indices_1, device);
        at::Tensor xla_values = bridge::CreateXlaTensor(values, device);
        at::Tensor xla_result = at::index_put(
            xla_params, {xla_indices_0, xla_indices_1}, xla_values, accumulate);
        AllClose(result, xla_result);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexPutHeadNull) {
  at::Tensor indices_0 =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  at::Tensor indices_null;
  at::Tensor indices_1 =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({4, 3, 3, 6, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {4, 3, 3, 6, 7}, at::TensorOptions(scalar_type));
    at::Tensor values = at::ones({3, 6, 7}, at::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      at::Tensor result = at::index_put(
          params, {indices_null, indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
        at::Tensor xla_indices_0 = bridge::CreateXlaTensor(indices_0, device);
        at::Tensor xla_indices_1 = bridge::CreateXlaTensor(indices_1, device);
        at::Tensor xla_values = bridge::CreateXlaTensor(values, device);
        at::Tensor xla_result = at::index_put(
            xla_params, {indices_null, xla_indices_0, xla_indices_1},
            xla_values, accumulate);
        AllClose(result, xla_result);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexPutMiddleNull) {
  at::Tensor indices_0 =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  at::Tensor indices_null;
  at::Tensor indices_1 =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({4, 3, 3, 6, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {4, 3, 3, 6, 7}, at::TensorOptions(scalar_type));
    at::Tensor values = at::ones({3, 6, 7}, at::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      at::Tensor result = at::index_put(
          params, {indices_0, indices_null, indices_1}, values, accumulate);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
        at::Tensor xla_indices_0 = bridge::CreateXlaTensor(indices_0, device);
        at::Tensor xla_indices_1 = bridge::CreateXlaTensor(indices_1, device);
        at::Tensor xla_values = bridge::CreateXlaTensor(values, device);
        at::Tensor xla_result = at::index_put(
            xla_params, {xla_indices_0, indices_null, xla_indices_1},
            xla_values, accumulate);
        AllClose(result, xla_result);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexPutTailNull) {
  at::Tensor indices_0 =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  at::Tensor indices_1 =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  at::Tensor indices_null;
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({4, 3, 3, 6, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {4, 3, 3, 6, 7}, at::TensorOptions(scalar_type));
    at::Tensor values = at::ones({3, 6, 7}, at::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      at::Tensor result = at::index_put(
          params, {indices_0, indices_1, indices_null}, values, accumulate);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
        at::Tensor xla_indices_0 = bridge::CreateXlaTensor(indices_0, device);
        at::Tensor xla_indices_1 = bridge::CreateXlaTensor(indices_1, device);
        at::Tensor xla_values = bridge::CreateXlaTensor(values, device);
        at::Tensor xla_result = at::index_put(
            xla_params, {xla_indices_0, xla_indices_1, indices_null},
            xla_values, accumulate);
        AllClose(result, xla_result);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexPutMiddleBroadcast) {
  at::Tensor indices_0 =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  at::Tensor indices_1 =
      at::randint(-3, 3, {2, 1, 3}, at::TensorOptions(at::kLong));
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({4, 3, 5, 6, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {4, 3, 5, 6, 7}, at::TensorOptions(scalar_type));
    at::Tensor values = at::ones({5, 6, 7}, at::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      at::Tensor result =
          at::index_put(params, {indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
        at::Tensor xla_indices_0 = bridge::CreateXlaTensor(indices_0, device);
        at::Tensor xla_indices_1 = bridge::CreateXlaTensor(indices_1, device);
        at::Tensor xla_values = bridge::CreateXlaTensor(values, device);
        at::Tensor xla_result = at::index_put(
            xla_params, {xla_indices_0, xla_indices_1}, xla_values, accumulate);
        AllClose(result, xla_result);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexPutTailBroadcast) {
  at::Tensor indices_0 =
      at::randint(-3, 3, {2, 1, 3}, at::TensorOptions(at::kLong));
  at::Tensor indices_1 =
      at::randint(-3, 3, {2, 1}, at::TensorOptions(at::kLong));
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({4, 3, 5, 6, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {4, 3, 5, 6, 7}, at::TensorOptions(scalar_type));
    at::Tensor values = at::ones({5, 6, 7}, at::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      at::Tensor result =
          at::index_put(params, {indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
        at::Tensor xla_indices_0 = bridge::CreateXlaTensor(indices_0, device);
        at::Tensor xla_indices_1 = bridge::CreateXlaTensor(indices_1, device);
        at::Tensor xla_values = bridge::CreateXlaTensor(values, device);
        at::Tensor xla_result = at::index_put(
            xla_params, {xla_indices_0, xla_indices_1}, xla_values, accumulate);
        AllClose(result, xla_result);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaskIndexPut) {
  at::Tensor indices = at::tensor({0, 1}, at::TensorOptions(at::kByte));
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor params =
        isFloatingType(scalar_type)
            ? at::rand({2, 2}, at::TensorOptions(scalar_type))
            : at::randint(100, {2, 2}, at::TensorOptions(scalar_type));
    at::Tensor values = at::ones({2}, at::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      at::Tensor result = at::index_put(params, {indices}, values, accumulate);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
        at::Tensor xla_indices = bridge::CreateXlaTensor(indices, device);
        at::Tensor xla_values = bridge::CreateXlaTensor(values, device);
        at::Tensor xla_result =
            at::index_put(xla_params, {xla_indices}, xla_values, accumulate);
        AllClose(result, xla_result);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexFillWithScalar) {
  at::Tensor index = at::tensor({0, 2}, at::TensorOptions(at::kLong));
  at::Scalar value = 42;
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor base =
        isFloatingType(scalar_type)
            ? at::rand({3, 4, 5}, at::TensorOptions(scalar_type))
            : at::randint(100, {3, 4, 5}, at::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      at::Tensor result = at::index_fill(base, dim, index, value);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_base = bridge::CreateXlaTensor(base, device);
        at::Tensor xla_index = bridge::CreateXlaTensor(index, device);
        at::Tensor xla_result = at::index_fill(xla_base, dim, xla_index, value);
        AllClose(result, xla_result);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexFillWithScalarInPlace) {
  at::Tensor index = at::tensor({0, 2}, at::TensorOptions(at::kLong));
  at::Scalar value = 42;
  int rank = 3;
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const Device& device) {
        at::Tensor base =
            isFloatingType(scalar_type)
                ? at::rand({3, 4, 5}, at::TensorOptions(scalar_type))
                : at::randint(100, {3, 4, 5}, at::TensorOptions(scalar_type));
        at::Tensor xla_base = bridge::CreateXlaTensor(base.clone(), device);
        at::Tensor result = base.index_fill_(dim, index, value);
        at::Tensor xla_index = bridge::CreateXlaTensor(index, device);
        at::Tensor xla_result = xla_base.index_fill_(dim, xla_index, value);
        AllClose(result, xla_result);
        AllClose(base, xla_base);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexFillWithTensor) {
  at::Tensor index = at::tensor({0, 2}, at::TensorOptions(at::kLong));
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor base =
        isFloatingType(scalar_type)
            ? at::rand({3, 4, 5}, at::TensorOptions(scalar_type))
            : at::randint(100, {3, 4, 5}, at::TensorOptions(scalar_type));
    at::Tensor value = at::scalar_tensor(42, at::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      at::Tensor result = at::index_fill(base, dim, index, value);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_base = bridge::CreateXlaTensor(base, device);
        at::Tensor xla_index = bridge::CreateXlaTensor(index, device);
        at::Tensor xla_value = bridge::CreateXlaTensor(value, device);
        at::Tensor xla_result =
            at::index_fill(xla_base, dim, xla_index, xla_value);
        AllClose(result, xla_result);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexFillWithTensorInPlace) {
  at::Tensor index = at::tensor({0, 2}, at::TensorOptions(at::kLong));
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor value = at::scalar_tensor(42, at::TensorOptions(scalar_type));
    int rank = 3;
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const Device& device) {
        at::Tensor base =
            isFloatingType(scalar_type)
                ? at::rand({3, 4, 5}, at::TensorOptions(scalar_type))
                : at::randint(100, {3, 4, 5}, at::TensorOptions(scalar_type));
        at::Tensor xla_base = bridge::CreateXlaTensor(base.clone(), device);
        at::Tensor result = base.index_fill_(dim, index, value);
        at::Tensor xla_index = bridge::CreateXlaTensor(index, device);
        at::Tensor xla_value = bridge::CreateXlaTensor(value, device);
        at::Tensor xla_result = xla_base.index_fill_(dim, xla_index, xla_value);
        AllClose(result, xla_result);
        AllClose(base, xla_base);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexAdd) {
  int index_size = 10;
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor base =
        isFloatingType(scalar_type)
            ? at::rand({5, 3, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {5, 3, 7}, at::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      at::Tensor index = at::randint(0, base.size(dim), {index_size},
                                     at::TensorOptions(at::kLong));
      std::vector<int64_t> value_sizes(base.sizes().begin(),
                                       base.sizes().end());
      int canonical_dim = dim < 0 ? dim + rank : dim;
      value_sizes[canonical_dim] = index_size;
      at::Tensor value =
          isFloatingType(scalar_type)
              ? at::rand(value_sizes, at::TensorOptions(scalar_type))
              : at::randint(100, value_sizes, at::TensorOptions(scalar_type));
      at::Tensor result = at::index_add(base, dim, index, value);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_base = bridge::CreateXlaTensor(base, device);
        at::Tensor xla_index = bridge::CreateXlaTensor(index, device);
        at::Tensor xla_value = bridge::CreateXlaTensor(value, device);
        at::Tensor xla_result =
            at::index_add(xla_base, dim, xla_index, xla_value);
        AllClose(result, xla_result);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexAddInPlace) {
  int index_size = 10;
  int rank = 3;
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const Device& device) {
        at::Tensor base =
            isFloatingType(scalar_type)
                ? at::rand({5, 3, 7}, at::TensorOptions(scalar_type))
                : at::randint(100, {5, 3, 7}, at::TensorOptions(scalar_type));
        at::Tensor index = at::randint(0, base.size(dim), {index_size},
                                       at::TensorOptions(at::kLong));
        std::vector<int64_t> value_sizes(base.sizes().begin(),
                                         base.sizes().end());
        int canonical_dim = dim < 0 ? dim + rank : dim;
        value_sizes[canonical_dim] = index_size;
        at::Tensor value =
            isFloatingType(scalar_type)
                ? at::rand(value_sizes, at::TensorOptions(scalar_type))
                : at::randint(100, value_sizes, at::TensorOptions(scalar_type));
        at::Tensor xla_base = bridge::CreateXlaTensor(base.clone(), device);
        at::Tensor result = base.index_add_(dim, index, value);
        at::Tensor xla_index = bridge::CreateXlaTensor(index, device);
        at::Tensor xla_value = bridge::CreateXlaTensor(value, device);
        at::Tensor xla_result = xla_base.index_add_(dim, xla_index, xla_value);
        AllClose(result, xla_result);
        AllClose(base, xla_base);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexCopy) {
  int index_size = 10;
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    at::Tensor base =
        isFloatingType(scalar_type)
            ? at::rand({5, 3, 7}, at::TensorOptions(scalar_type))
            : at::randint(100, {5, 3, 7}, at::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      at::Tensor index = at::randint(0, base.size(dim), {index_size},
                                     at::TensorOptions(at::kLong));
      std::vector<int64_t> value_sizes(base.sizes().begin(),
                                       base.sizes().end());
      int canonical_dim = dim < 0 ? dim + rank : dim;
      value_sizes[canonical_dim] = index_size;
      at::Tensor value =
          isFloatingType(scalar_type)
              ? at::rand(value_sizes, at::TensorOptions(scalar_type))
              : at::randint(100, value_sizes, at::TensorOptions(scalar_type));
      at::Tensor result = at::index_copy(base, dim, index, value);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_base = bridge::CreateXlaTensor(base, device);
        at::Tensor xla_index = bridge::CreateXlaTensor(index, device);
        at::Tensor xla_value = bridge::CreateXlaTensor(value, device);
        at::Tensor xla_result =
            at::index_copy(xla_base, dim, xla_index, xla_value);
        AllClose(result, xla_result);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexCopyInPlace) {
  int index_size = 10;
  int rank = 3;
  for (at::ScalarType scalar_type :
       {at::kFloat, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const Device& device) {
        at::Tensor base =
            isFloatingType(scalar_type)
                ? at::rand({5, 3, 7}, at::TensorOptions(scalar_type))
                : at::randint(100, {5, 3, 7}, at::TensorOptions(scalar_type));
        at::Tensor index = at::randint(0, base.size(dim), {index_size},
                                       at::TensorOptions(at::kLong));
        std::vector<int64_t> value_sizes(base.sizes().begin(),
                                         base.sizes().end());
        int canonical_dim = dim < 0 ? dim + rank : dim;
        value_sizes[canonical_dim] = index_size;
        at::Tensor value =
            isFloatingType(scalar_type)
                ? at::rand(value_sizes, at::TensorOptions(scalar_type))
                : at::randint(100, value_sizes, at::TensorOptions(scalar_type));
        at::Tensor xla_base = bridge::CreateXlaTensor(base.clone(), device);
        at::Tensor result = base.index_copy_(dim, index, value);
        at::Tensor xla_index = bridge::CreateXlaTensor(index, device);
        at::Tensor xla_value = bridge::CreateXlaTensor(value, device);
        at::Tensor xla_result = xla_base.index_copy_(dim, xla_index, xla_value);
        AllClose(result, xla_result);
        AllClose(base, xla_base);
      });
    }
  }
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

TEST_F(AtenXlaTensorTest, TestReluInPlace) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor output = at::relu_(input);
    at::Tensor xla_output = at::relu_(xla_input);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestHardshrink) {
  at::Tensor input = at::randn({100}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::hardshrink(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::hardshrink(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestSoftshrink) {
  at::Tensor input = at::randn({100}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::softshrink(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::softshrink(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestHardtanh) {
  at::Tensor input = at::randn({100}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::hardtanh(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::hardtanh(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestHardtanhInPlace) {
  at::Tensor input = at::randn({100}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor output = at::hardtanh_(input);
    at::Tensor xla_output = at::hardtanh_(xla_input);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestLeakyRelu) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  double negative_slope = 0.01;
  at::Tensor output = at::leaky_relu(input, negative_slope);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::leaky_relu(xla_input, negative_slope);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestLeakyReluInPlace) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  double negative_slope = 0.01;
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor output = at::leaky_relu_(input, negative_slope);
    at::Tensor xla_output = at::leaky_relu_(xla_input, negative_slope);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
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

TEST_F(AtenXlaTensorTest, TestExpm1) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::expm1(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::expm1(xla_a);
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

TEST_F(AtenXlaTensorTest, TestLog2) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::log2(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::log2(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestLog10) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::log10(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::log10(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestLog1p) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::log1p(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::log1p(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestErf) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::erf(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::erf(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestErfc) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::erfc(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::erfc(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestErfinv) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::erfinv(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::erfinv(xla_a);
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

TEST_F(AtenXlaTensorTest, TestRsqrt) {
  at::Tensor a = at::abs(at::rand({2, 2}, at::TensorOptions(at::kFloat)));
  at::Tensor b = at::rsqrt(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::rsqrt(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestReciprocal) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::reciprocal(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::reciprocal(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestPowTensorScalar) {
  at::Tensor base = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar exponent = 4.09;
  at::Tensor result = at::pow(base, exponent);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_base = bridge::CreateXlaTensor(base, device);
    at::Tensor xla_result = at::pow(xla_base, exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestPowTensorScalarInPlace) {
  at::Tensor base = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar exponent = 4.09;
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_base = bridge::CreateXlaTensor(base.clone(), device);
    at::Tensor result = base.pow_(exponent);
    at::Tensor xla_result = xla_base.pow_(exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(base, xla_base, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestPowTensorTensor) {
  at::Tensor base = at::abs(at::rand({4, 2}, at::TensorOptions(at::kFloat)));
  at::Tensor exponent = at::rand({4, 2});
  at::Tensor result = at::pow(base, exponent);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_base = bridge::CreateXlaTensor(base, device);
    at::Tensor xla_exponent = bridge::CreateXlaTensor(exponent, device);
    at::Tensor xla_result = at::pow(xla_base, xla_exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestPowTensorTensorInPlace) {
  at::Tensor base = at::abs(at::rand({4, 2}, at::TensorOptions(at::kFloat)));
  at::Tensor exponent = at::rand({4, 2});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_base = bridge::CreateXlaTensor(base.clone(), device);
    at::Tensor result = base.pow_(exponent);
    at::Tensor xla_exponent = bridge::CreateXlaTensor(exponent, device);
    at::Tensor xla_result = xla_base.pow_(xla_exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(base, xla_base, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestPowTensorTensorBroadcast) {
  at::Tensor base = at::abs(at::rand({4, 2}, at::TensorOptions(at::kFloat)));
  at::Tensor exponent = at::rand({4, 1});
  at::Tensor result = at::pow(base, exponent);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_base = bridge::CreateXlaTensor(base, device);
    at::Tensor xla_exponent = bridge::CreateXlaTensor(exponent, device);
    at::Tensor xla_result = at::pow(xla_base, xla_exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestPowScalarTensor) {
  at::Scalar base = 3.5;
  at::Tensor exponent = at::rand({4, 2});
  at::Tensor result = at::pow(base, exponent);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_exponent = bridge::CreateXlaTensor(exponent, device);
    at::Tensor xla_result = at::pow(base, xla_exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestFmodScalar) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Scalar divisor = 2.0;
  at::Tensor b = at::fmod(a, divisor);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::fmod(xla_a, divisor);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestFmodScalarInPlace) {
  at::Scalar divisor = 2.0;
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor b = a.fmod_(divisor);
    at::Tensor xla_b = xla_a.fmod_(divisor);
    AllClose(b, xla_b);
    AllClose(a, xla_a);
  });
}

TEST_F(AtenXlaTensorTest, TestFmodTensor) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 10.0;
  at::Tensor c = at::fmod(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::fmod(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestFmodTensorInPlace) {
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 10.0;
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor c = a.fmod_(b);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = xla_a.fmod_(xla_b);
    AllClose(c, xla_c);
    AllClose(a, xla_a);
  });
}

TEST_F(AtenXlaTensorTest, TestRemainderScalar) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Scalar divisor = -2.0;
  at::Tensor b = at::remainder(a, divisor);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::remainder(xla_a, divisor);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestRemainderScalarInPlace) {
  at::Scalar divisor = -2.0;
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor b = a.remainder_(divisor);
    at::Tensor xla_b = xla_a.remainder_(divisor);
    AllClose(b, xla_b);
    AllClose(a, xla_a);
  });
}

TEST_F(AtenXlaTensorTest, TestRemainderTensor) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::randn({2, 2}, at::TensorOptions(at::kFloat)) * 10.0;
  at::Tensor c = at::remainder(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::remainder(xla_a, xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-4, /*atol=*/1e-6);
  });
}

TEST_F(AtenXlaTensorTest, TestRemainderTensorInPlace) {
  at::Tensor b = at::randn({2, 2}, at::TensorOptions(at::kFloat)) * 10.0;
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor c = a.remainder_(b);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = xla_a.remainder_(xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-4, /*atol=*/1e-6);
    AllClose(a, xla_a, /*rtol=*/1e-4, /*atol=*/1e-6);
  });
}

TEST_F(AtenXlaTensorTest, TestWhere) {
  at::Tensor a = at::rand({3, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({3, 3}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::empty({3, 3}, at::TensorOptions(at::kByte));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = i == j;
    }
  }
  at::Tensor d = at::where(c, a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
    at::Tensor xla_d = at::where(xla_c, xla_a, xla_b);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenXlaTensorTest, TestWhereBroadcast) {
  at::Tensor a = at::rand({3, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::zeros({}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::empty({3, 3}, at::TensorOptions(at::kByte));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = i == j;
    }
  }
  at::Tensor d = at::where(c, a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
    at::Tensor xla_d = at::where(xla_c, xla_a, xla_b);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenXlaTensorTest, TestWhereAutograd) {
  at::Tensor a = at::rand({3, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({3, 3}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::empty({3, 3}, at::TensorOptions(at::kByte));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = i == j;
    }
  }
  at::Tensor d = at::_s_where(c, a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
    at::Tensor xla_d = at::_s_where(xla_c, xla_a, xla_b);
    AllClose(d, xla_d);
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

TEST_F(AtenXlaTensorTest, TestThresholdInPlace) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  at::Tensor output = input.clone();
  float threshold = 0.4;
  float value = 20;
  at::threshold_(output, threshold, value);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_output = bridge::CreateXlaTensor(input, device);
    at::threshold_(xla_output, threshold, value);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestElu) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  at::Scalar alpha = 0.5;
  at::Scalar scale = 2.5;
  at::Scalar input_scale = 1.5;
  float value = 20;
  at::Tensor output = at::elu(input, alpha, scale, input_scale);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::elu(xla_input, alpha, scale, input_scale);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestEluInPlace) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  at::Scalar alpha = 0.5;
  at::Scalar scale = 2.5;
  at::Scalar input_scale = 1.5;
  float value = 20;
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor output = at::elu_(input, alpha, scale, input_scale);
    at::Tensor xla_output = at::elu_(xla_input, alpha, scale, input_scale);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestSelu) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::selu(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::selu(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestSeluInPlace) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor output = at::selu_(input);
    at::Tensor xla_output = at::selu_(xla_input);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestCelu) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  at::Scalar alpha = 2.5;
  at::Tensor output = at::celu(input, alpha);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::celu(xla_input, alpha);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestCeluInPlace) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  at::Scalar alpha = 2.5;
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor output = at::celu_(input, alpha);
    at::Tensor xla_output = at::celu_(xla_input, alpha);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
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

TEST_F(AtenXlaTensorTest, TestEmbedding) {
  at::Tensor a = at::rand({32, 4}, at::TensorOptions(at::kFloat));
  at::Tensor i = at::randint(0, 31, {3, 4}, at::TensorOptions(at::kLong));
  at::Tensor b =
      at::embedding(a, i, /*padding_idx=*/0, /*scale_grad_by_freq=*/false,
                    /*sparse=*/false);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_i = bridge::CreateXlaTensor(i, device);
    at::Tensor xla_b = at::embedding(xla_a, xla_i, /*padding_idx=*/0,
                                     /*scale_grad_by_freq=*/false,
                                     /*sparse=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestOneHot) {
  int num_classes = 5;
  at::Tensor input =
      at::randint(0, num_classes, {10}, at::TensorOptions(at::kLong));
  at::Tensor output = at::one_hot(input, num_classes);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::one_hot(xla_input, num_classes);
    EXPECT_TRUE(EqualValues(output, xla_output));
  });
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

TEST_F(AtenXlaTensorTest, TestTransposeInPlace) {
  at::Tensor input = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor output = input.t_();
    at::Tensor xla_output = xla_input.t_();
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestReshape) {
  at::Tensor input = at::rand({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::reshape(input, {-1, 320});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::reshape(xla_input, {-1, 320});
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

TEST_F(AtenXlaTensorTest, TestViewSqueezeAddInPlace) {
  at::Tensor input = at::zeros({2, 3, 1}, at::TensorOptions(at::kFloat));
  std::vector<int64_t> view_size = {2, 3, 1, 1};
  int squeeze_dim = 2;
  at::Tensor one = at::tensor(1.0, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor output = input.view(view_size);
    output.squeeze_(squeeze_dim);
    output.add_(one, 1.0);
    at::Tensor xla_one = bridge::CreateXlaTensor(one, device);
    at::Tensor xla_output = xla_input.view(view_size);
    xla_output.squeeze_(squeeze_dim);
    xla_output.add_(xla_one, 1.0);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestUnsafeView) {
  at::Tensor input = at::rand({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::_unsafe_view(input, {-1, 320});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::_unsafe_view(xla_input, {-1, 320});
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestNarrow) {
  at::Tensor a = at::rand({8, 10, 4, 4}, at::TensorOptions(at::kFloat));
  for (xla::int64 dim : {1, -3}) {
    for (xla::int64 start : {2, -8}) {
      at::Tensor b = a.narrow(dim, start, 6);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
        at::Tensor xla_b = xla_a.narrow(dim, start, 6);
        AllClose(b, xla_b);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestNarrowUpdate) {
  for (xla::int64 dim : {1, -2}) {
    for (xla::int64 start : {2, -6}) {
      at::Tensor a = at::rand({3, 8, 3}, at::TensorOptions(at::kFloat));
      at::Tensor a_copy = a.clone();
      at::Tensor b = at::rand({3, 4, 3}, at::TensorOptions(at::kFloat));
      at::Tensor c = a.narrow(dim, start, 4);
      c.add_(b, 1.0);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_a = bridge::CreateXlaTensor(a_copy, device);
        at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
        at::Tensor xla_c = xla_a.narrow(dim, start, 4);
        xla_c.add_(xla_b, 1.0);
        AllClose(c, xla_c);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestNarrowUpdateBaseCheck) {
  for (xla::int64 dim : {0, -2}) {
    for (xla::int64 start : {2, -6}) {
      at::Tensor a = at::zeros({8, 3}, at::TensorOptions(at::kFloat));
      at::Tensor a_copy = a.clone();
      at::Tensor b = at::ones({4, 3}, at::TensorOptions(at::kFloat));
      at::Tensor c = a.narrow(dim, start, 4);
      c.add_(b, 1.0);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_a = bridge::CreateXlaTensor(a_copy, device);
        at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
        at::Tensor xla_c = xla_a.narrow(dim, start, 4);
        xla_c.add_(xla_b, 1.0);
        AllClose(a, xla_a);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestNarrowUpdateTwoSlices) {
  for (xla::int64 dim : {0, -2}) {
    for (xla::int64 start0 : {2, -6}) {
      for (xla::int64 start1 : {6, -2}) {
        at::Tensor a = at::zeros({8, 3}, at::TensorOptions(at::kFloat));
        at::Tensor a_copy = a.clone();
        at::Tensor b = at::ones({2, 3}, at::TensorOptions(at::kFloat));
        at::Tensor c = b + 1;
        at::Tensor d = a.narrow(dim, start0, 2);
        at::Tensor e = a.narrow(dim, start1, 2);
        d.add_(b, 1.0);
        e.add_(c, 1.0);
        ForEachDevice([&](const Device& device) {
          at::Tensor xla_a = bridge::CreateXlaTensor(a_copy, device);
          at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
          at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
          at::Tensor xla_d = xla_a.narrow(dim, start0, 2);
          at::Tensor xla_e = xla_a.narrow(dim, start1, 2);
          xla_d.add_(xla_b, 1.0);
          xla_e.add_(xla_c, 1.0);
          AllClose(d, xla_d);
          AllClose(e, xla_e);
          AllClose(a, xla_a);
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestNarrowUpdateView) {
  for (xla::int64 dim : {0, -3}) {
    for (xla::int64 start : {2, -6}) {
      at::Tensor a = at::rand({8, 2, 3}, at::TensorOptions(at::kFloat));
      at::Tensor a_copy = a.clone();
      at::Tensor b = at::rand({4, 6}, at::TensorOptions(at::kFloat));
      at::Tensor c = a.narrow(dim, start, 4);
      at::Tensor d = c.view({4, 6});
      d.add_(b, 1.0);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_a = bridge::CreateXlaTensor(a_copy, device);
        at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
        at::Tensor xla_c = xla_a.narrow(dim, start, 4);
        at::Tensor xla_d = xla_c.view({4, 6});
        xla_d.add_(xla_b, 1.0);
        AllClose(d, xla_d);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestNarrowInNarrowUpdate) {
  for (xla::int64 dim : {1, -2}) {
    for (xla::int64 start0 : {1, -7}) {
      for (xla::int64 start1 : {1, -5}) {
        at::Tensor a = at::rand({3, 8, 3}, at::TensorOptions(at::kFloat));
        at::Tensor a_copy = a.clone();
        at::Tensor b = at::rand({3, 2, 3}, at::TensorOptions(at::kFloat));
        at::Tensor c = a.narrow(dim, start0, 6);
        at::Tensor d = c.narrow(dim, start1, 2);
        d.add_(b, 1.0);
        ForEachDevice([&](const Device& device) {
          at::Tensor xla_a = bridge::CreateXlaTensor(a_copy, device);
          at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
          at::Tensor xla_c = xla_a.narrow(dim, start0, 6);
          at::Tensor xla_d = xla_c.narrow(dim, start1, 2);
          xla_d.add_(xla_b, 1.0);
          AllClose(a, xla_a);
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestNarrowCopy) {
  for (xla::int64 dim : {1, -3}) {
    for (xla::int64 start : {2, -8}) {
      ForEachDevice([&](const Device& device) {
        at::Tensor input =
            at::rand({8, 10, 4, 4}, at::TensorOptions(at::kFloat));
        at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
        at::Tensor result = input.narrow_copy(dim, start, 6);
        input.add_(1);
        at::Tensor xla_result = xla_input.narrow_copy(dim, start, 6);
        xla_input.add_(1);
        AllClose(result, xla_result);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestViewAs) {
  at::Tensor input = at::rand({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor empty = at::empty({32, 320});
  at::Tensor output = input.view_as(empty);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_empty = bridge::CreateXlaTensor(empty, device);
    at::Tensor xla_output = xla_input.view_as(xla_empty);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestLogSoftmax) {
  at::Tensor input = at::rand({5, 3, 4, 2}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
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
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      at::Tensor output = at::softmax(input, dim);
      at::Tensor xla_output = at::softmax(xla_input, dim);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestSoftmaxWrapper) {
  at::Tensor input = at::rand({10, 8, 24, 16}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      at::Tensor output = at::_softmax(input, dim, /*half_to_float=*/false);
      at::Tensor xla_output =
          at::_softmax(xla_input, dim, /*half_to_float=*/false);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestSoftplus) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::softplus(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::softplus(xla_input);
    AllClose(output, xla_output, /*rtol=*/1e-4);
  });
}

TEST_F(AtenXlaTensorTest, TestMaxPool1D) {
  at::Tensor input = at::rand({1, 64, 112}, at::TensorOptions(at::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          at::Tensor output =
              at::max_pool1d(input, /*kernel_size=*/{kernel_size},
                             /*stride=*/{stride},
                             /*padding=*/{padding}, /*dilation=*/{dilation},
                             /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output =
                at::max_pool1d(xla_input,
                               /*kernel_size=*/{kernel_size},
                               /*stride=*/{stride},
                               /*padding=*/{padding},
                               /*dilation=*/{dilation},
                               /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
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

TEST_F(AtenXlaTensorTest, TestMaxPool3D) {
  at::Tensor input =
      at::rand({1, 64, 16, 16, 16}, at::TensorOptions(at::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          at::Tensor output = at::max_pool3d(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output = at::max_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool3DIncompleteAttributes) {
  at::Tensor input =
      at::rand({1, 64, 16, 16, 16}, at::TensorOptions(at::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          at::Tensor output =
              at::max_pool3d(input, /*kernel_size=*/{kernel_size},
                             /*stride=*/{},
                             /*padding=*/{padding},
                             /*dilation=*/{dilation, dilation, dilation},
                             /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output =
                at::max_pool3d(xla_input,
                               /*kernel_size=*/{kernel_size},
                               /*stride=*/{},
                               /*padding=*/{padding},
                               /*dilation=*/{dilation, dilation, dilation},
                               /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool3DNonSquare) {
  at::Tensor input =
      at::rand({1, 64, 16, 16, 16}, at::TensorOptions(at::kFloat));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          at::Tensor output = at::max_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
              /*stride=*/{stride, stride + 1, stride},
              /*padding=*/{padding, padding + 1, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output = at::max_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
                /*stride=*/{stride, stride + 1, stride},
                /*padding=*/{padding, padding + 1, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAvgPool1D) {
  at::Tensor input = at::rand({4, 1, 28}, at::TensorOptions(at::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          at::Tensor output =
              at::avg_pool1d(input, /*kernel_size=*/{kernel_size},
                             /*stride=*/{stride},
                             /*padding=*/{padding}, /*ceil_mode=*/ceil_mode,
                             /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output =
                at::avg_pool1d(xla_input,
                               /*kernel_size=*/{kernel_size},
                               /*stride=*/{stride},
                               /*padding=*/{padding},
                               /*ceil_mode=*/ceil_mode,
                               /*count_include_pad=*/count_include_pad);
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

TEST_F(AtenXlaTensorTest, TestAvgPool3D) {
  at::Tensor input =
      at::rand({4, 1, 28, 28, 28}, at::TensorOptions(at::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          at::Tensor output = at::avg_pool3d(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output = at::avg_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAvgPool3DIncompleteAttributes) {
  at::Tensor input =
      at::rand({4, 1, 28, 28, 28}, at::TensorOptions(at::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          at::Tensor output =
              at::avg_pool3d(input, /*kernel_size=*/{kernel_size},
                             /*stride=*/{},
                             /*padding=*/{padding}, /*ceil_mode=*/ceil_mode,
                             /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output =
                at::avg_pool3d(xla_input,
                               /*kernel_size=*/{kernel_size},
                               /*stride=*/{},
                               /*padding=*/{padding},
                               /*ceil_mode=*/ceil_mode,
                               /*count_include_pad=*/count_include_pad);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAvgPool3DNonSquare) {
  at::Tensor input =
      at::rand({4, 1, 28, 28, 28}, at::TensorOptions(at::kFloat));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          at::Tensor output = at::avg_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
              /*stride=*/{stride, stride + 1, stride},
              /*padding=*/{padding, padding + 1, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output = at::avg_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
                /*stride=*/{stride, stride + 1, stride},
                /*padding=*/{padding, padding + 1, padding},
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

TEST_F(AtenXlaTensorTest, TestTransposedConv2D) {
  int in_channels = 3;
  int out_channels = 7;
  int kernel_size = 5;
  at::Tensor input =
      at::rand({4, out_channels, 28, 28}, at::TensorOptions(at::kFloat));
  at::Tensor weight =
      at::rand({out_channels, in_channels, kernel_size, kernel_size},
               at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({in_channels}, at::TensorOptions(at::kFloat));
  at::Tensor bias_undef;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (int dilation = 1; dilation <= 2; ++dilation) {
        for (int output_padding = 0;
             output_padding < std::min(stride, dilation); ++output_padding) {
          for (bool with_bias : {true, false}) {
            // Test dilation through the CPU interop.
            at::Tensor output = at::conv_transpose2d(
                input, weight, with_bias ? bias : bias_undef,
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*output_padding=*/output_padding,
                /*groups=*/1,
                /*dilation=*/{dilation, dilation});
            ForEachDevice([&](const Device& device) {
              at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
              at::Tensor xla_weight = bridge::CreateXlaTensor(weight, device);
              at::Tensor xla_bias = bridge::CreateXlaTensor(bias, device);
              at::Tensor xla_output = at::conv_transpose2d(
                  xla_input, xla_weight, with_bias ? xla_bias : bias_undef,
                  /*stride=*/{stride, stride},
                  /*padding=*/{padding, padding},
                  /*output_padding=*/output_padding,
                  /*groups=*/1,
                  /*dilation=*/{dilation, dilation});
              AllClose(output, xla_output);
            });
          }
        };
      }
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
  at::Tensor input = at::rand({batch, classes}, at::TensorOptions(at::kFloat));
  at::Tensor target =
      at::randint(0, classes, {batch}, at::TensorOptions(at::kLong));
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

TEST_F(AtenXlaTensorTest, TestSmoothL1Loss) {
  at::Tensor input = at::randn({2, 4}, at::TensorOptions(at::kFloat));
  at::Tensor target = at::randn({2, 4}, at::TensorOptions(at::kFloat));
  for (Reduction::Reduction reduction :
       {Reduction::None, Reduction::Mean, Reduction::Sum}) {
    at::Tensor output = at::smooth_l1_loss(input, target, reduction);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_target = bridge::CreateXlaTensor(target, device);
      at::Tensor xla_output =
          at::smooth_l1_loss(xla_input, xla_target, reduction);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestBatchNorm1D) {
  int num_features = 3;
  at::Tensor input =
      at::rand({14, num_features, 7}, at::TensorOptions(at::kFloat));
  at::Tensor weight = at::rand({num_features}, at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({num_features}, at::TensorOptions(at::kFloat));
  at::Tensor running_mean =
      at::zeros({num_features}, at::TensorOptions(at::kFloat));
  at::Tensor running_var =
      at::ones({num_features}, at::TensorOptions(at::kFloat));
  double momentum = 0.1;
  double eps = 0.5;
  at::Tensor undef;
  for (bool training : {true, false}) {
    for (bool undef_weight_bias : {false, true}) {
      at::Tensor output = at::batch_norm(
          /*input=*/input, /*weight=*/undef_weight_bias ? undef : weight,
          /*bias=*/undef_weight_bias ? undef : bias,
          /*running_mean=*/running_mean, /*running_var=*/running_var,
          /*training=*/training, /*momentum=*/momentum, /*eps=*/eps,
          /*cudnn_enabled=*/false);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
        at::Tensor xla_weight =
            undef_weight_bias ? undef : bridge::CreateXlaTensor(weight, device);
        at::Tensor xla_bias =
            undef_weight_bias ? undef : bridge::CreateXlaTensor(bias, device);
        at::Tensor xla_running_mean =
            bridge::CreateXlaTensor(running_mean, device);
        at::Tensor xla_running_var =
            bridge::CreateXlaTensor(running_var, device);
        at::Tensor xla_output = at::batch_norm(
            /*input=*/xla_input, /*weight=*/xla_weight, /*bias=*/xla_bias,
            /*running_mean=*/xla_running_mean, /*running_var=*/xla_running_var,
            /*training=*/training, /*momentum=*/momentum, /*eps=*/eps,
            /*cudnn_enabled=*/false);
        AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestBatchNorm2D) {
  int num_features = 3;
  at::Tensor input =
      at::rand({14, num_features, 5, 7}, at::TensorOptions(at::kFloat));
  at::Tensor weight = at::rand({num_features}, at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({num_features}, at::TensorOptions(at::kFloat));
  at::Tensor running_mean =
      at::zeros({num_features}, at::TensorOptions(at::kFloat));
  at::Tensor running_var =
      at::ones({num_features}, at::TensorOptions(at::kFloat));
  double momentum = 0.1;
  double eps = 0.5;
  at::Tensor undef;
  for (bool training : {true, false}) {
    for (bool undef_weight_bias : {false, true}) {
      at::Tensor output = at::batch_norm(
          /*input=*/input, /*weight=*/undef_weight_bias ? undef : weight,
          /*bias=*/undef_weight_bias ? undef : bias,
          /*running_mean=*/running_mean, /*running_var=*/running_var,
          /*training=*/training, /*momentum=*/momentum, /*eps=*/eps,
          /*cudnn_enabled=*/false);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
        at::Tensor xla_weight =
            undef_weight_bias ? undef : bridge::CreateXlaTensor(weight, device);
        at::Tensor xla_bias =
            undef_weight_bias ? undef : bridge::CreateXlaTensor(bias, device);
        at::Tensor xla_running_mean =
            bridge::CreateXlaTensor(running_mean, device);
        at::Tensor xla_running_var =
            bridge::CreateXlaTensor(running_var, device);
        at::Tensor xla_output = at::batch_norm(
            /*input=*/xla_input, /*weight=*/xla_weight, /*bias=*/xla_bias,
            /*running_mean=*/xla_running_mean, /*running_var=*/xla_running_var,
            /*training=*/training, /*momentum=*/momentum, /*eps=*/eps,
            /*cudnn_enabled=*/false);
        AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestDim) {
  at::Tensor input = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    EXPECT_EQ(input.dim(), xla_input.dim());
  });
}

TEST_F(AtenXlaTensorTest, TestContiguous) {
  at::Tensor input = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::native::contiguous(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::native::contiguous(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestSqueezeAll) {
  at::Tensor input = at::rand({2, 1, 3, 1}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::squeeze(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::squeeze(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestSqueezeAllInPlace) {
  ForEachDevice([&](const Device& device) {
    at::Tensor input = at::rand({2, 1, 3, 1}, at::TensorOptions(at::kFloat));
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor output = input.squeeze_();
    at::Tensor xla_output = xla_input.squeeze_();
    AllClose(output, xla_output);
    AllClose(input, xla_input);
    ASSERT_EQ(input.dim(), xla_input.dim());
    for (int64_t dim_idx = 0; dim_idx < input.dim(); ++dim_idx) {
      ASSERT_EQ(input.size(dim_idx), xla_input.size(dim_idx));
    }
  });
}

TEST_F(AtenXlaTensorTest, TestSqueezeOne) {
  at::Tensor input = at::rand({2, 1, 3, 1}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor output = at::squeeze(input, dim);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::squeeze(xla_input, dim);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestSqueezeOneInPlace) {
  int rank = 4;
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const Device& device) {
      at::Tensor input = at::rand({2, 1, 3, 1}, at::TensorOptions(at::kFloat));
      at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
      at::Tensor output = input.squeeze_(dim);
      at::Tensor xla_output = xla_input.squeeze_(dim);
      AllClose(output, xla_output);
      AllClose(input, xla_input);
      ASSERT_EQ(input.dim(), xla_input.dim());
      for (int64_t dim_idx = 0; dim_idx < input.dim(); ++dim_idx) {
        ASSERT_EQ(input.size(dim_idx), xla_input.size(dim_idx));
      }
    });
  }
}

TEST_F(AtenXlaTensorTest, TestUnsqueeze) {
  at::Tensor input = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  int rank = input.dim() + 1;
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor output = at::unsqueeze(input, dim);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::unsqueeze(xla_input, dim);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestUnsqueezeInPlace) {
  at::Tensor input = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  int rank = input.dim() + 1;
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
      at::Tensor output = input.unsqueeze_(dim);
      at::Tensor xla_output = xla_input.unsqueeze_(dim);
      AllClose(output, xla_output);
      AllClose(input, xla_input);
      ASSERT_EQ(input.dim(), xla_input.dim());
      for (int64_t dim_idx = 0; dim_idx < input.dim(); ++dim_idx) {
        ASSERT_EQ(input.size(dim_idx), xla_input.size(dim_idx));
      }
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMaskedFill) {
  at::Tensor input = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor mask = at::randint(0, 2, {2, 3}, at::TensorOptions(at::kByte));
  at::Scalar value(42);
  at::Tensor result = at::masked_fill(input, mask, value);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_mask = bridge::CreateXlaTensor(mask, device);
    at::Tensor xla_result = at::masked_fill(xla_input, xla_mask, value);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestMaskedFillInPlace) {
  at::Scalar value(42);
  at::Tensor mask = at::randint(0, 2, {2, 3}, at::TensorOptions(at::kByte));
  ForEachDevice([&](const Device& device) {
    at::Tensor input = at::rand({2, 3}, at::TensorOptions(at::kFloat));
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor xla_mask = bridge::CreateXlaTensor(mask, device);
    at::Tensor result = input.masked_fill_(mask, value);
    at::Tensor xla_result = xla_input.masked_fill_(xla_mask, value);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestMaskedFillBroadcast) {
  at::Tensor input = at::rand({2, 5, 4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor mask = at::randint(0, 2, {4, 1}, at::TensorOptions(at::kByte));
  at::Scalar value(42);
  at::Tensor result = at::masked_fill(input, mask, value);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_mask = bridge::CreateXlaTensor(mask, device);
    at::Tensor xla_result = at::masked_fill(xla_input, xla_mask, value);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestFill) {
  at::Scalar value(42);
  ForEachDevice([&](const Device& device) {
    at::Tensor input = at::empty({2, 3}, at::TensorOptions(at::kFloat));
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor result = at::fill_(input, value);
    at::Tensor xla_result = at::fill_(xla_input, value);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestFillWithRank0) {
  at::Tensor value = at::scalar_tensor(42);
  ForEachDevice([&](const Device& device) {
    at::Tensor input = at::empty({2, 3}, at::TensorOptions(at::kFloat));
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor result = at::fill_(input, value);
    at::Tensor xla_value = bridge::CreateXlaTensor(value, device);
    at::Tensor xla_result = at::fill_(xla_input, value);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestPermute) {
  at::Tensor input = at::rand({2, 3, 4}, at::TensorOptions(at::kFloat));
  std::vector<std::vector<int64_t>> dims_permutations = {
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  int rank = input.dim();
  for (std::vector<int64_t> dims_permutation : dims_permutations) {
    for (bool negative_dims : {false, true}) {
      if (negative_dims) {
        std::for_each(dims_permutation.begin(), dims_permutation.end(),
                      [rank](int64_t& dim) { dim -= rank; });
      }
      at::Tensor output = input.permute(dims_permutation);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
        at::Tensor xla_output = xla_input.permute(dims_permutation);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestPermuteMod) {
  std::vector<std::vector<int64_t>> dims_permutations = {
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  std::vector<int64_t> input_sizes = {2, 3, 4};
  int rank = input_sizes.size();
  for (std::vector<int64_t> dims_permutation : dims_permutations) {
    for (bool negative_dims : {false, true}) {
      if (negative_dims) {
        std::for_each(dims_permutation.begin(), dims_permutation.end(),
                      [rank](int64_t& dim) { dim -= rank; });
      }
      at::Tensor input = at::zeros(input_sizes, at::TensorOptions(at::kFloat));
      at::Tensor one = at::tensor(1.0, at::TensorOptions(at::kFloat));
      at::Tensor output = input.permute(dims_permutation);
      output.add_(one, 1.0);
      input.add_(one, 1.0);
      ForEachDevice([&](const Device& device) {
        at::Tensor xinput =
            at::zeros(input_sizes, at::TensorOptions(at::kFloat));
        at::Tensor xla_input = bridge::CreateXlaTensor(xinput, device);
        at::Tensor xla_one = bridge::CreateXlaTensor(one, device);
        at::Tensor xla_output = xla_input.permute(dims_permutation);
        xla_output.add_(xla_one, 1.0);
        xla_input.add_(xla_one, 1.0);
        AllClose(output, xla_output);
        AllClose(input, xla_input);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestFlip) {
  at::Tensor input = at::rand({2, 3, 4}, at::TensorOptions(at::kFloat));
  std::vector<std::vector<int64_t>> dim_powerset = {
      {0}, {1}, {2}, {0, 1}, {1, 2}, {2, 0}, {0, 1, 2}};
  for (std::vector<int64_t> flip_dims : dim_powerset) {
    for (bool negative_dims : {false, true}) {
      if (negative_dims) {
        std::for_each(flip_dims.begin(), flip_dims.end(),
                      [](int64_t& dim) { dim -= 3; });
      }
      at::Tensor output = at::flip(input, flip_dims);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
        at::Tensor xla_output = at::flip(xla_input, flip_dims);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestPixelShuffle) {
  at::Tensor input = at::rand({5, 18, 4, 4}, at::TensorOptions(at::kFloat));
  int upscale_factor = 3;
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor output = at::pixel_shuffle(input, upscale_factor);
    at::Tensor xla_output = at::pixel_shuffle(xla_input, upscale_factor);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestSumToSize) {
  at::Tensor input = at::rand({4, 6, 3, 7}, at::TensorOptions(at::kFloat));
  std::vector<int64_t> out_size = {4, 1, 1, 7};
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor output = input.sum_to_size(out_size);
    at::Tensor xla_output = xla_input.sum_to_size(out_size);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestTransposeDims) {
  at::Tensor input = at::rand({2, 3, 4}, at::TensorOptions(at::kFloat));
  int dim0 = 0;
  int dim1 = 2;
  at::Tensor output = at::transpose(input, dim0, dim1);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::transpose(xla_input, dim0, dim1);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestTransposeDimsMod) {
  std::vector<int64_t> input_sizes = {2, 3, 4};
  int dim0 = 0;
  int dim1 = 2;
  at::Tensor input = at::zeros(input_sizes, at::TensorOptions(at::kFloat));
  at::Tensor one = at::tensor(1.0, at::TensorOptions(at::kFloat));
  at::Tensor output = at::transpose(input, dim0, dim1);
  output.add_(one, 1.0);
  input.add_(one, 1.0);
  ForEachDevice([&](const Device& device) {
    at::Tensor xinput = at::zeros(input_sizes, at::TensorOptions(at::kFloat));
    at::Tensor xla_input = bridge::CreateXlaTensor(xinput, device);
    at::Tensor xla_one = bridge::CreateXlaTensor(one, device);
    at::Tensor xla_output = at::transpose(xla_input, dim0, dim1);
    xla_output.add_(xla_one, 1.0);
    xla_input.add_(xla_one, 1.0);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestTransposeDimsInPlace) {
  at::Tensor input = at::rand({2, 3, 4}, at::TensorOptions(at::kFloat));
  int dim0 = 0;
  int dim1 = 2;
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor output = input.transpose_(dim0, dim1);
    at::Tensor xla_output = xla_input.transpose_(dim0, dim1);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestSplit) {
  at::Tensor input = at::rand({7, 8, 9}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  for (int split_size : {2, 3}) {
    for (int dim = -rank; dim < rank; ++dim) {
      std::vector<at::Tensor> outputs = at::split(input, split_size, dim);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
        std::vector<at::Tensor> xla_outputs =
            at::split(xla_input, split_size, dim);
        ASSERT_EQ(outputs.size(), xla_outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
          AllClose(outputs[i], xla_outputs[i]);
        }
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestSplitEmpty) {
  at::Tensor input = at::rand({0}, at::TensorOptions(at::kFloat));
  int split_size = 0;
  int dim = 0;
  std::vector<at::Tensor> outputs = at::split(input, split_size, dim);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    std::vector<at::Tensor> xla_outputs = at::split(xla_input, split_size, dim);
    ASSERT_EQ(outputs.size(), xla_outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      AllClose(outputs[i], xla_outputs[i]);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestSplitWithSizes) {
  at::Tensor input = at::rand({15, 15, 15}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    std::vector<at::Tensor> outputs =
        at::split_with_sizes(input, {4, 5, 6}, dim);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      std::vector<at::Tensor> xla_outputs =
          at::split_with_sizes(xla_input, {4, 5, 6}, dim);
      ASSERT_EQ(outputs.size(), xla_outputs.size());
      for (size_t i = 0; i < outputs.size(); ++i) {
        AllClose(outputs[i], xla_outputs[i]);
      }
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCrossImplicitDim) {
  std::vector<std::vector<int64_t>> dim_sizes = {
      {4, 5, 3}, {4, 3, 5}, {3, 4, 5}};
  for (auto dim_size : dim_sizes) {
    at::Tensor input = at::rand(dim_size, at::TensorOptions(at::kFloat));
    at::Tensor other = at::rand(dim_size, at::TensorOptions(at::kFloat));
    at::Tensor result = at::cross(input, other);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_other = bridge::CreateXlaTensor(other, device);
      at::Tensor xla_result = at::cross(xla_input, xla_other);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCrossExplicitDim) {
  std::vector<int64_t> dim_size = {3, 3};
  at::Tensor input = at::rand(dim_size, at::TensorOptions(at::kFloat));
  at::Tensor other = at::rand(dim_size, at::TensorOptions(at::kFloat));
  int rank = dim_size.size();
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor result = at::cross(input, other, dim);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_other = bridge::CreateXlaTensor(other, device);
      at::Tensor xla_result = at::cross(xla_input, xla_other, dim);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestTriu) {
  int size = 5;
  at::Tensor input = at::rand({size, size}, at::TensorOptions(at::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::triu(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::triu(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestTriuNonSquare) {
  int size = 5;
  at::Tensor input = at::rand({size, size + 1}, at::TensorOptions(at::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::triu(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::triu(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestTriuBatch) {
  int size = 5;
  int batch_size = 3;
  at::Tensor input =
      at::rand({batch_size, size, size}, at::TensorOptions(at::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::triu(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::triu(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestTril) {
  int size = 5;
  at::Tensor input = at::rand({size, size}, at::TensorOptions(at::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::tril(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::tril(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestTrilNonSquare) {
  int size = 5;
  at::Tensor input = at::rand({size, size + 1}, at::TensorOptions(at::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::tril(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::tril(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestTrilBatch) {
  int size = 5;
  int batch_size = 3;
  at::Tensor input =
      at::rand({batch_size, size, size}, at::TensorOptions(at::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::tril(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::tril(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestTriuInPlace) {
  int size = 5;
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    ForEachDevice([&](const Device& device) {
      at::Tensor input = at::rand({size, size}, at::TensorOptions(at::kFloat));
      at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
      at::Tensor output = input.triu_(diagonal);
      at::Tensor xla_output = xla_input.triu_(diagonal);
      AllClose(output, xla_output);
      AllClose(input, xla_input);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestTrilInPlace) {
  int size = 5;
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    ForEachDevice([&](const Device& device) {
      at::Tensor input = at::rand({size, size}, at::TensorOptions(at::kFloat));
      at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
      at::Tensor output = input.tril_(diagonal);
      at::Tensor xla_output = xla_input.tril_(diagonal);
      AllClose(output, xla_output);
      AllClose(input, xla_input);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestTrace) {
  int n = 5;
  at::Tensor input = at::rand({n, n}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::trace(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::trace(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestTraceWide) {
  int lines = 3;
  int cols = 5;
  at::Tensor input = at::rand({lines, cols}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::trace(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::trace(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestTraceNarrow) {
  int lines = 5;
  int cols = 3;
  at::Tensor input = at::rand({lines, cols}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::trace(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::trace(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestDiagRank1) {
  int size = 7;
  at::Tensor input = at::rand({size}, at::TensorOptions(at::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -2 * size; diagonal <= 2 * size; ++diagonal) {
    at::Tensor output = at::diag(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::diag(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestDiagRank2) {
  int size = 7;
  at::Tensor input = at::rand({size, size}, at::TensorOptions(at::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::diag(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::diag(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestDiagFlat) {
  at::Tensor input = at::rand({4, 3, 6, 7}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  for (int diagonal = -10; diagonal < 10; ++diagonal) {
    at::Tensor output = at::diagflat(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::diagflat(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestDiagonal) {
  int size = 5;
  at::Tensor input = at::rand({size, size}, at::TensorOptions(at::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::diagonal(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::diagonal(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestDiagonalNonSquare) {
  int size = 5;
  at::Tensor input = at::rand({size, size + 1}, at::TensorOptions(at::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::diagonal(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::diagonal(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestDiagonalBatch) {
  int size = 5;
  int batch_size = 3;
  int dim1 = 1;
  int dim2 = 2;
  at::Tensor input =
      at::rand({batch_size, size, size}, at::TensorOptions(at::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output =
        at::diagonal(input, diagonal, /*dim1=*/dim1, /*dim1=*/dim2);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output =
          at::diagonal(xla_input, diagonal, /*dim1=*/dim1, /*dim1=*/dim2);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestFlatten) {
  at::Tensor input = at::rand({4, 7, 5, 3});
  int rank = input.dim();
  for (int pos_start_dim = 0; pos_start_dim < rank; ++pos_start_dim) {
    for (int pos_end_dim = pos_start_dim; pos_end_dim < rank; ++pos_end_dim) {
      for (bool negative_start_dim : {false, true}) {
        for (bool negative_end_dim : {false, true}) {
          int start_dim =
              negative_start_dim ? pos_start_dim - rank : pos_start_dim;
          int end_dim = negative_end_dim ? pos_end_dim - rank : pos_end_dim;
          at::Tensor output = at::flatten(input, start_dim, end_dim);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output = at::flatten(xla_input, start_dim, end_dim);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestBitwiseAnd) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor rhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor result = lhs.__and__(rhs);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs, device);
    at::Tensor xla_rhs = bridge::CreateXlaTensor(rhs, device);
    at::Tensor xla_result = xla_lhs.__and__(xla_rhs);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseAndInPlace) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor rhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs.clone(), device);
    at::Tensor result = lhs.__iand__(rhs);
    at::Tensor xla_rhs = bridge::CreateXlaTensor(rhs, device);
    at::Tensor xla_result = xla_lhs.__iand__(xla_rhs);
    AllClose(result, xla_result);
    AllClose(lhs, xla_lhs);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseAndScalar) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Scalar rhs(123456789);
  at::Tensor result = lhs.__and__(rhs);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs, device);
    at::Tensor xla_result = xla_lhs.__and__(rhs);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseAndScalarInPlace) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Scalar rhs(123456789);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs.clone(), device);
    at::Tensor result = lhs.__iand__(rhs);
    at::Tensor xla_result = xla_lhs.__iand__(rhs);
    AllClose(result, xla_result);
    AllClose(lhs, xla_lhs);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseAndPromotion) {
  at::Tensor input = at::rand({4, 2}, at::TensorOptions(at::kFloat));
  at::Tensor view = input.reshape(-1);
  at::Tensor result = at::__and__(view.gt(0), view.ne(0));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = torch::autograd::make_variable(
        bridge::CreateXlaTensor(input, device), false);
    at::Tensor xla_view = xla_input.reshape(-1);
    at::Tensor xla_result = at::__and__(xla_view.gt(0), xla_view.ne(0));
    EXPECT_TRUE(EqualValues(result, xla_result));
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseOr) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor rhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor result = lhs.__or__(rhs);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs, device);
    at::Tensor xla_rhs = bridge::CreateXlaTensor(rhs, device);
    at::Tensor xla_result = xla_lhs.__or__(xla_rhs);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseOrInPlace) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor rhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs.clone(), device);
    at::Tensor result = lhs.__ior__(rhs);
    at::Tensor xla_rhs = bridge::CreateXlaTensor(rhs, device);
    at::Tensor xla_result = xla_lhs.__ior__(xla_rhs);
    AllClose(result, xla_result);
    AllClose(lhs, xla_lhs);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseOrScalar) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Scalar rhs(123456789);
  at::Tensor result = lhs.__or__(rhs);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs, device);
    at::Tensor xla_result = xla_lhs.__or__(rhs);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseOrScalarInPlace) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Scalar rhs(123456789);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs.clone(), device);
    at::Tensor result = lhs.__ior__(rhs);
    at::Tensor xla_result = xla_lhs.__ior__(rhs);
    AllClose(result, xla_result);
    AllClose(lhs, xla_lhs);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseXor) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor rhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor result = lhs.__xor__(rhs);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs, device);
    at::Tensor xla_rhs = bridge::CreateXlaTensor(rhs, device);
    at::Tensor xla_result = xla_lhs.__xor__(xla_rhs);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseXorInPlace) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor rhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs.clone(), device);
    at::Tensor result = lhs.__ixor__(rhs);
    at::Tensor xla_rhs = bridge::CreateXlaTensor(rhs, device);
    at::Tensor xla_result = xla_lhs.__ixor__(xla_rhs);
    AllClose(result, xla_result);
    AllClose(lhs, xla_lhs);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseXorScalar) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Scalar rhs(123456789);
  at::Tensor result = lhs.__xor__(rhs);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs, device);
    at::Tensor xla_result = xla_lhs.__xor__(rhs);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseXorScalarInPlace) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Scalar rhs(123456789);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs.clone(), device);
    at::Tensor result = lhs.__ixor__(rhs);
    at::Tensor xla_result = xla_lhs.__ixor__(rhs);
    AllClose(result, xla_result);
    AllClose(lhs, xla_lhs);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseAndAutograd) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor rhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor result = at::legacy::th::__and__(lhs, rhs);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = torch::autograd::make_variable(
        bridge::CreateXlaTensor(lhs, device), false);
    at::Tensor xla_rhs = torch::autograd::make_variable(
        bridge::CreateXlaTensor(rhs, device), false);
    at::Tensor xla_result = at::legacy::th::__and__(xla_lhs, xla_rhs);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseOrAutograd) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor rhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor result = at::legacy::th::__or__(lhs, rhs);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = torch::autograd::make_variable(
        bridge::CreateXlaTensor(lhs, device), false);
    at::Tensor xla_rhs = torch::autograd::make_variable(
        bridge::CreateXlaTensor(rhs, device), false);
    at::Tensor xla_result = at::legacy::th::__or__(xla_lhs, xla_rhs);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseXorAutograd) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor rhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor result = at::legacy::th::__xor__(lhs, rhs);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = torch::autograd::make_variable(
        bridge::CreateXlaTensor(lhs, device), false);
    at::Tensor xla_rhs = torch::autograd::make_variable(
        bridge::CreateXlaTensor(rhs, device), false);
    at::Tensor xla_result = at::legacy::th::__xor__(xla_lhs, xla_rhs);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestLshift) {
  at::Tensor input = at::randn({4, 2}, at::TensorOptions(at::kFloat));
  at::Tensor shift_amount = at::randint(16, input.sizes());
  at::Tensor result = at::__lshift__(input, shift_amount);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_shift_amount = bridge::CreateXlaTensor(shift_amount, device);
    at::Tensor xla_result = at::__lshift__(xla_input, xla_shift_amount);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestLshiftInPlace) {
  at::Tensor input = at::randn({4, 2}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor shift_amount = at::randint(16, input.sizes());
    at::Tensor result = input.__ilshift__(shift_amount);
    at::Tensor xla_shift_amount = bridge::CreateXlaTensor(shift_amount, device);
    at::Tensor xla_result = xla_input.__ilshift__(xla_shift_amount);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestLshiftScalar) {
  at::Tensor input = at::randn({4, 2}, at::TensorOptions(at::kFloat));
  at::Scalar shift_amount = 3;
  at::Tensor result = at::__lshift__(input, shift_amount);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::__lshift__(xla_input, shift_amount);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestLshiftScalarInPlace) {
  at::Tensor input = at::randn({4, 2}, at::TensorOptions(at::kFloat));
  at::Scalar shift_amount = 3;
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor result = input.__ilshift__(shift_amount);
    at::Tensor xla_result = xla_input.__ilshift__(shift_amount);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestRshift) {
  at::Tensor input = at::randn({4, 2}, at::TensorOptions(at::kFloat));
  at::Tensor shift_amount = at::randint(16, input.sizes());
  at::Tensor result = at::__rshift__(input, shift_amount);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_shift_amount = bridge::CreateXlaTensor(shift_amount, device);
    at::Tensor xla_result = at::__rshift__(xla_input, xla_shift_amount);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestRshiftInPlace) {
  at::Tensor input = at::randn({4, 2}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor shift_amount = at::randint(16, input.sizes());
    at::Tensor result = input.__irshift__(shift_amount);
    at::Tensor xla_shift_amount = bridge::CreateXlaTensor(shift_amount, device);
    at::Tensor xla_result = xla_input.__irshift__(xla_shift_amount);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestRshiftScalar) {
  at::Tensor input = at::randn({4, 2}, at::TensorOptions(at::kFloat));
  at::Scalar shift_amount = 3;
  at::Tensor result = at::__rshift__(input, shift_amount);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::__rshift__(xla_input, shift_amount);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestRshiftScalarInPlace) {
  at::Tensor input = at::randn({4, 2}, at::TensorOptions(at::kFloat));
  at::Scalar shift_amount = 3;
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor result = input.__irshift__(shift_amount);
    at::Tensor xla_result = xla_input.__irshift__(shift_amount);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestMeshgrid) {
  at::Tensor a = at::rand({3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::rand({4}, at::TensorOptions(at::kFloat));
  auto d = at::meshgrid({a, b, c});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
    auto xla_d = at::meshgrid({xla_a, xla_b, xla_c});
    EXPECT_EQ(d.size(), xla_d.size());
    for (size_t i = 0; i < d.size(); ++i) {
      AllClose(d[i], xla_d[i]);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestConstantPad) {
  at::Tensor input = at::rand({4, 2, 5}, at::TensorOptions(at::kFloat));
  std::vector<int64_t> pad{1, 2, 3, 4, 5, 6};
  float pad_value = 5;
  at::Tensor output = at::constant_pad_nd(input, pad, pad_value);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::constant_pad_nd(xla_input, pad, pad_value);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestConstantPadIncomplete) {
  at::Tensor input = at::rand({4, 2, 5}, at::TensorOptions(at::kFloat));
  std::vector<int64_t> pad{1, 2};
  float pad_value = 5;
  at::Tensor output = at::constant_pad_nd(input, pad, pad_value);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::constant_pad_nd(xla_input, pad, pad_value);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestAsStrided) {
  at::Tensor input = at::rand({128, 320}, at::TensorOptions(at::kFloat));
  std::vector<int64_t> size = {128, 20, 4, 4};
  std::vector<int64_t> stride = {320, 16, 4, 1};
  at::Tensor output = at::as_strided(input, /*size=*/size, /*stride=*/stride);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output =
        at::as_strided(xla_input, /*size=*/size, /*stride=*/stride);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestAsStridedInPlace) {
  at::Tensor input = at::rand({128, 320}, at::TensorOptions(at::kFloat));
  std::vector<int64_t> size = {128, 20, 4, 4};
  std::vector<int64_t> stride = {320, 16, 4, 1};
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor output =
        at::as_strided_(input, /*size=*/size, /*stride=*/stride);
    at::Tensor xla_output =
        at::as_strided_(xla_input, /*size=*/size, /*stride=*/stride);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestAsStridedWithOffset) {
  at::Tensor input = at::rand({4, 8, 2}, at::TensorOptions(at::kFloat));
  std::vector<int64_t> size = {4, 4, 2};
  std::vector<int64_t> stride = {8, 2, 1};
  int64_t storage_offset = 4;
  at::Tensor output = at::as_strided(input, /*size=*/size, /*stride=*/stride,
                                     /*storage_offset=*/storage_offset);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output =
        at::as_strided(xla_input, /*size=*/size, /*stride=*/stride,
                       /*storage_offset=*/storage_offset);
    AllClose(output, xla_output);
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
            TestBackward(
                {at::rand({4, 1, 28, 28}, at::TensorOptions(at::kFloat))},
                device, testfn);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAvgPool3DBackward) {
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          auto testfn =
              [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
            return at::avg_pool3d(
                inputs[0],
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
          };

          ForEachDevice([&](const Device& device) {
            TestBackward(
                {at::rand({4, 1, 28, 28, 28}, at::TensorOptions(at::kFloat))},
                device, testfn);
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
      TestBackward({at::rand({4, 1, 28, 28}, at::TensorOptions(at::kFloat))},
                   device, testfn);
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
                with_bias
                    ? at::rand({out_channels}, at::TensorOptions(at::kFloat))
                    : at::Tensor();
            TestBackward(
                {at::rand({4, in_channels, 32, 32},
                          at::TensorOptions(at::kFloat)),
                 at::rand({out_channels, in_channels, kernel_size, kernel_size},
                          at::TensorOptions(at::kFloat)),
                 bias},
                device, testfn);
          });
        }
      };
    }
  }
}

TEST_F(AtenXlaTensorTest, TestTransposedConv2DBackward) {
  int in_channels = 2;
  int out_channels = 3;
  int kernel_size = 5;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (int dilation = 1; dilation <= 2; ++dilation) {
        for (int output_padding = 0;
             output_padding < std::min(stride, dilation); ++output_padding) {
          for (bool with_bias : {true, false}) {
            // Test dilation through the CPU interop.
            auto testfn =
                [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
              return at::conv_transpose2d(inputs[0], inputs[1], inputs[2],
                                          /*stride=*/{stride, stride},
                                          /*padding=*/{padding, padding},
                                          /*output_padding=*/output_padding,
                                          /*groups=*/1,
                                          /*dilation=*/{dilation, dilation});
            };
            ForEachDevice([&](const Device& device) {
              at::Tensor input = at::rand({4, out_channels, 14, 14},
                                          at::TensorOptions(at::kFloat));
              at::Tensor weight = at::rand(
                  {out_channels, in_channels, kernel_size, kernel_size},
                  at::TensorOptions(at::kFloat));
              at::Tensor bias =
                  with_bias
                      ? at::rand({in_channels}, at::TensorOptions(at::kFloat))
                      : at::Tensor();
              TestBackward({input, weight, bias}, device, testfn);
            });
          }
        };
      }
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
          TestBackward(
              {at::rand({1, 64, 112, 112}, at::TensorOptions(at::kFloat))},
              device, testfn);
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool3DBackward) {
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
          return at::max_pool3d(
              inputs[0],
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding}, /*dilation=*/{1, 1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const Device& device) {
          TestBackward(
              {at::rand({1, 64, 16, 16, 16}, at::TensorOptions(at::kFloat))},
              device, testfn);
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestTanhBackward) {
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::tanh(inputs[0]);
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({at::rand({2, 2}, at::TensorOptions(at::kFloat))}, device,
                 testfn, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestSigmoidBackward) {
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::sigmoid(inputs[0]);
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({at::rand({2, 2}, at::TensorOptions(at::kFloat))}, device,
                 testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestLogSigmoidBackward) {
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::log_sigmoid(inputs[0]);
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({at::rand({2, 2}, at::TensorOptions(at::kFloat))}, device,
                 testfn, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestLogSoftmaxBackward) {
  for (int dim = -4; dim < 4; ++dim) {
    auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
      return at::log_softmax(inputs[0], dim);
    };

    ForEachDevice([&](const Device& device) {
      TestBackward({at::rand({5, 3, 4, 2}, at::TensorOptions(at::kFloat))},
                   device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestSoftmaxBackward) {
  for (int dim = -4; dim < 4; ++dim) {
    auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
      return at::softmax(inputs[0], dim);
    };

    ForEachDevice([&](const Device& device) {
      TestBackward({at::rand({5, 3, 4, 2}, at::TensorOptions(at::kFloat))},
                   device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestSoftplusBackward) {
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::softplus(inputs[0]);
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat))},
                 device, testfn, /*rtol=*/1e-4);
  });
}

TEST_F(AtenXlaTensorTest, TestReluBackward) {
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::relu(inputs[0]);
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat))},
                 device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestHardshrinkBackward) {
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::hardshrink(inputs[0]);
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({at::randn({100}, at::TensorOptions(at::kFloat))}, device,
                 testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestSoftshrinkBackward) {
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::softshrink(inputs[0]);
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({at::randn({100}, at::TensorOptions(at::kFloat))}, device,
                 testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestHardtanhBackward) {
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::hardtanh(inputs[0]);
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({at::randn({100}, at::TensorOptions(at::kFloat))}, device,
                 testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestEluBackward) {
  at::Scalar alpha = 0.5;
  at::Scalar scale = 2.5;
  at::Scalar input_scale = 1.5;
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::elu(inputs[0], alpha, scale, input_scale);
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat))},
                 device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestLeakyReluBackward) {
  double negative_slope = 0.01;
  auto testfn = [=](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::leaky_relu(inputs[0], negative_slope);
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat))},
                 device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestTransposeBackward) {
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::t(inputs[0]);
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({at::rand({2, 3}, at::TensorOptions(at::kFloat))}, device,
                 testfn);
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
          {at::rand({labels}, at::TensorOptions(at::kFloat)),
           at::rand({in_channels, out_channels}, at::TensorOptions(at::kFloat)),
           at::rand({out_channels, labels}, at::TensorOptions(at::kFloat))},
          device, testfn);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestNllLossBackward) {
  int batch = 3;
  int classes = 5;
  at::Tensor input = at::rand({batch, classes}, at::TensorOptions(at::kFloat));
  at::Tensor target =
      at::randint(0, classes, {batch}, at::TensorOptions(at::kLong));
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

TEST_F(AtenXlaTensorTest, TestSmoothL1LossBackward) {
  at::Tensor input = at::randn({2, 4}, at::TensorOptions(at::kFloat));
  at::Tensor target = at::randn({2, 4}, at::TensorOptions(at::kFloat));
  for (Reduction::Reduction reduction :
       {Reduction::None, Reduction::Mean, Reduction::Sum}) {
    auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
      return at::smooth_l1_loss(/*input=*/inputs[0], /*target=*/inputs[1],
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
    TestBackward({at::rand({32, 20, 4, 4}, at::TensorOptions(at::kFloat))},
                 device, testfn);
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
  at::Tensor undef;
  for (bool undef_weight_bias : {false, true}) {
    ForEachDevice([&](const Device& device) {
      at::Tensor input =
          at::rand({14, num_features, 5, 7}, at::TensorOptions(at::kFloat));
      at::Tensor weight =
          undef_weight_bias
              ? undef
              : at::rand({num_features}, at::TensorOptions(at::kFloat));
      at::Tensor bias =
          undef_weight_bias
              ? undef
              : at::rand({num_features}, at::TensorOptions(at::kFloat));
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
}

TEST_F(AtenXlaTensorTest, TestBCEWithLogitsBackward) {
  int batch = 10;
  int classes = 5;
  at::Tensor undef;
  for (Reduction::Reduction reduction :
       {Reduction::None, Reduction::Mean, Reduction::Sum}) {
    auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
      return at::binary_cross_entropy_with_logits(
          /*input=*/inputs[0], /*target=*/inputs[1], /*weight=*/inputs[2],
          /*pos_weight=*/inputs[3],
          /*reduction=*/reduction);
    };
    for (bool undef_weight : {false, true}) {
      for (bool undef_pos_weight : {false, true}) {
        at::Tensor input =
            at::rand({batch, classes}, at::TensorOptions(at::kFloat));
        at::Tensor target =
            at::rand({batch, classes}, at::TensorOptions(at::kFloat));
        at::Tensor weight =
            undef_weight ? undef
                         : at::rand({classes}, at::TensorOptions(at::kFloat));
        at::Tensor pos_weight =
            undef_pos_weight
                ? undef
                : at::rand({classes}, at::TensorOptions(at::kFloat));
        ForEachDevice([&](const Device& device) {
          TestBackward({input, target, weight, pos_weight}, device, testfn,
                       /*rtol=*/1e-3, /*atol=*/1e-5,
                       /*inputs_require_grad=*/{true, true, false, false});
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestKlDivBackward) {
  at::Tensor input = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor target = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  for (Reduction::Reduction reduction : {Reduction::Mean, Reduction::Sum}) {
    auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
      return at::kl_div(/*self=*/inputs[0], /*target=*/inputs[1], reduction);
    };
    ForEachDevice([&](const Device& device) {
      TestBackward({input, target}, device, testfn, /*rtol=*/1e-4,
                   /*atol=*/1e-5);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestEmbeddingBackward) {
  int num_weights = 32;
  for (int padding_idx = -1; padding_idx < num_weights; ++padding_idx) {
    for (bool scale_grad_by_freq : {false, true}) {
      auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
        return at::embedding(inputs[0], inputs[1], /*padding_idx=*/padding_idx,
                             /*scale_grad_by_freq=*/scale_grad_by_freq,
                             /*sparse=*/false);
      };
      ForEachDevice([&](const Device& device) {
        at::Tensor weight =
            at::rand({num_weights, 7}, at::TensorOptions(at::kFloat));
        at::Tensor indices =
            at::randint(num_weights, {3, 9, 4}, at::TensorOptions(at::kLong));
        TestBackward({weight, indices}, device, testfn, /*rtol=*/1e-5,
                     /*atol=*/1e-8, {true, false});
      });
    }
  }
}

}  // namespace cpp_test
}  // namespace torch_xla
