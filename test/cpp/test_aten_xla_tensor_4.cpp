#include <gtest/gtest.h>
#include <torch/torch.h>

#include <iostream>

#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "test/cpp/cpp_test_util.h"
#include "test/cpp/torch_xla_test.h"
#include "third_party/xla_client/metrics.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ops/dynamic_ir.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace cpp_test {
namespace {

class AtenXlaTensorTest : public AtenXlaTensorTestBase {};

}  // namespace

TEST_F(AtenXlaTensorTest, TestScalarTensor) {
  torch::Tensor scalar_tensor =
      torch::scalar_tensor(1., torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_scalar_tensor = torch::scalar_tensor(
        1., torch::TensorOptions(torch::kFloat).device(torch::kXLA));
    AllClose(scalar_tensor, xla_scalar_tensor);
  });
}

TEST_F(AtenXlaTensorTest, TestClone) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = xla_a.clone();
    AllClose(a, xla_b);
    xla_a.add_(1.0);
    AllClose(a, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestTo) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    AllClose(a, xla_a);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_copy_from", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestIsFloatingPoint) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    bool is_float = torch::is_floating_point(a);
    bool xla_is_float = torch::is_floating_point(xla_a);
    EXPECT_EQ(is_float, xla_is_float);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  // This check only checks scalar_type which doesn't call into XLA.
  // So there's no positive asserts.
}

TEST_F(AtenXlaTensorTest, TestIsSigned) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    bool is_signed = torch::is_signed(a);
    bool xla_is_signed = torch::is_signed(xla_a);
    EXPECT_EQ(is_signed, xla_is_signed);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  // This check only checks scalar_type which doesn't call into XLA.
  // So there's no positive asserts.
}

TEST_F(AtenXlaTensorTest, TestCastByte) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::_cast_Byte(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::_cast_Byte(xla_a);
    AllEqual(b, xla_b);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestCastChar) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::_cast_Char(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::_cast_Char(xla_a);
    AllEqual(b, xla_b);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestCastShort) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::_cast_Short(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::_cast_Short(xla_a);
    AllEqual(b, xla_b);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestCastInt) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::_cast_Int(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::_cast_Int(xla_a);
    AllEqual(b, xla_b);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestCastLong) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::_cast_Long(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::_cast_Long(xla_a);
    AllEqual(b, xla_b);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestCastFloat) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::_cast_Float(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::_cast_Float(xla_a);
    AllEqual(b, xla_b);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestRetainType) {
  torch::Tensor xla_a = torch::zeros(
      {2, 2}, torch::TensorOptions(torch::kByte).device(torch::kXLA));
  torch::Tensor xla_b = torch::ones(
      {2, 2}, torch::TensorOptions(torch::kByte).device(torch::kXLA));
  torch::Tensor xla_c = xla_a + xla_b;
  EXPECT_EQ(xla_c.scalar_type(), torch::ScalarType::Byte);
}

TEST_F(AtenXlaTensorTest, TestAdd) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::add(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestAddInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor c = a.add_(b);
    torch::Tensor xla_c = xla_a.add_(xla_b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestAddScalar) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar b(1);
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_c = torch::add(xla_a, b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestAddScalarInPlace) {
  torch::Scalar b(1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor c = a.add_(b);
    torch::Tensor xla_c = xla_a.add_(b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestAddZeroSizeDim) {
  torch::Tensor a = torch::rand({0, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({1, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::add(xla_a, xla_b);
    AllClose(c, xla_c);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSub) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::sub(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::sub(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestSubInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor c = a.sub_(b);
    torch::Tensor xla_c = xla_a.sub_(xla_b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestSubScalar) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar b(1);
  torch::Tensor c = torch::sub(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_c = torch::sub(xla_a, b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestSubScalarInPlace) {
  torch::Scalar b(1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor c = a.sub_(b);
    torch::Tensor xla_c = xla_a.sub_(b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestSymSizes) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    ASSERT_EQ(*a.sym_sizes().at(0).maybe_as_int(), 2);

    torch::Tensor b = torch::tensor({{0.0, 1.0}, {0.0, 0.0}},
                                    torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_b = CopyToDevice(b, device);
    xla_b = torch::nonzero(xla_b);
    auto s0 = xla_b.sym_sizes().at(0);
    ASSERT_FALSE(s0.maybe_as_int().has_value());
    auto sininode = dynamic_cast<XLASymNodeImpl*>(s0.toSymNodeImplUnowned());
    auto snode =
        std::dynamic_pointer_cast<torch_xla::SizeNode>(sininode->node());
    ASSERT_TRUE(snode);
    ASSERT_EQ(snode->getStaticValue(), 4);
    ASSERT_EQ(snode->getDynamicValue(), 1);
  });
}

TEST_F(AtenXlaTensorTest, TestMul) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::mul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::mul(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMulInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor c = a.mul_(b);
    torch::Tensor xla_c = xla_a.mul_(xla_b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMulScalar) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar b(3);
  torch::Tensor c = torch::mul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_c = torch::mul(xla_a, b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMulScalarInPlace) {
  torch::Scalar b(3);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor c = a.mul_(b);
    torch::Tensor xla_c = xla_a.mul_(b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestDiv) {
  for (torch::ScalarType scalar_type1 :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor a =
        isFloatingType(scalar_type1)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
            : torch::randint(0, 100, {3, 4},
                             torch::TensorOptions(scalar_type1));
    for (torch::ScalarType scalar_type2 :
         {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
          torch::kLong}) {
      torch::Tensor b =
          isFloatingType(scalar_type2)
              ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
              : torch::randint(1, 100, {3, 4},
                               torch::TensorOptions(scalar_type2));
      torch::Tensor c = torch::div(a, b);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::Tensor xla_b = CopyToDevice(b, device);
        torch::Tensor xla_c = torch::div(xla_a, xla_b);
        AllClose(c, xla_c);
      });
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::div", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestDivWithRoundingMode) {
  c10::optional<c10::string_view> rounding_modes[] = {"trunc", "floor",
                                                      c10::nullopt};
  for (const auto& rounding_mode : rounding_modes) {
    for (torch::ScalarType scalar_type1 :
         {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
          torch::kLong}) {
      int lower_bound = (scalar_type1 == torch::kByte) ? 0 : -100;
      torch::Tensor a =
          isFloatingType(scalar_type1)
              ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
              : torch::randint(lower_bound, 50, {3, 4},
                               torch::TensorOptions(scalar_type1));
      for (torch::ScalarType scalar_type2 :
           {torch::kFloat, torch::kByte, torch::kChar, torch::kShort,
            torch::kInt, torch::kLong}) {
        torch::Tensor b =
            isFloatingType(scalar_type2)
                ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
                : torch::randint(51, 100, {3, 4},
                                 torch::TensorOptions(scalar_type2));
        torch::Tensor c = torch::div(a, b, rounding_mode);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          torch::Tensor xla_b = CopyToDevice(b, device);
          torch::Tensor xla_c = torch::div(xla_a, xla_b, rounding_mode);
          AllClose(c, xla_c);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::div", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestDivInPlace) {
  for (torch::ScalarType scalar_type1 : {torch::kFloat}) {
    torch::Tensor a =
        isFloatingType(scalar_type1)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
            : torch::randint(0, 100, {3, 4},
                             torch::TensorOptions(scalar_type1));
    for (torch::ScalarType scalar_type2 : {torch::kFloat}) {
      torch::Tensor b =
          isFloatingType(scalar_type2)
              ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
              : torch::randint(1, 100, {3, 4},
                               torch::TensorOptions(scalar_type2));
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::Tensor c = a.div_(b);
        torch::Tensor xla_b = CopyToDevice(b, device);
        torch::Tensor xla_c = xla_a.div_(xla_b);
        ;
        AllClose(c, xla_c);
      });
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::div", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestDivInPlaceWithRoundingMode) {
  c10::optional<c10::string_view> rounding_modes[] = {"trunc", "floor",
                                                      c10::nullopt};
  for (const auto& rounding_mode : rounding_modes) {
    for (torch::ScalarType scalar_type1 : {torch::kFloat}) {
      torch::Tensor a =
          isFloatingType(scalar_type1)
              ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
              : torch::randint(-100, 100, {3, 4},
                               torch::TensorOptions(scalar_type1));
      for (torch::ScalarType scalar_type2 : {torch::kFloat}) {
        torch::Tensor b =
            isFloatingType(scalar_type2)
                ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
                : torch::randint(1, 100, {3, 4},
                                 torch::TensorOptions(scalar_type2));
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          torch::Tensor c = a.div_(b, rounding_mode);
          torch::Tensor xla_b = CopyToDevice(b, device);
          torch::Tensor xla_c = xla_a.div_(xla_b, rounding_mode);
          AllClose(c, xla_c);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::div", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestDivScalar) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor a =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type))
            : torch::randint(1, 100, {3, 4}, torch::TensorOptions(scalar_type));
    for (bool is_float : {true, false}) {
      torch::Scalar b = is_float ? torch::Scalar(3.0) : torch::Scalar(3);
      torch::Tensor c = torch::div(a, b);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::Tensor xla_c = torch::div(xla_a, b);
        AllClose(c, xla_c);
      });
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::div", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestDivScalarInPlace) {
  for (torch::ScalarType scalar_type : {torch::kFloat}) {
    torch::Tensor a =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type))
            : torch::randint(1, 100, {3, 4}, torch::TensorOptions(scalar_type));
    for (bool is_float : {true, false}) {
      torch::Scalar b = is_float ? torch::Scalar(3.0) : torch::Scalar(3);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::Tensor c = a.div_(b);
        torch::Tensor xla_c = xla_a.div_(b);
        AllClose(c, xla_c);
      });
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::div", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestDivOut) {
  for (torch::ScalarType scalar_type : {torch::kFloat, torch::kDouble}) {
    torch::Tensor a = torch::rand({3, 4}, torch::TensorOptions(scalar_type));
    torch::Tensor b = torch::rand({3, 4}, torch::TensorOptions(scalar_type));
    torch::Tensor c = torch::empty({3, 4}, torch::TensorOptions(scalar_type));
    torch::div_out(c, a, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = torch::empty({3, 4}, xla_b.options());
      torch::div_out(xla_c, xla_a, xla_b);
      AllClose(c, xla_c);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::div", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestRsubScalar) {
  torch::Tensor input =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar other(1.5);
  torch::Scalar alpha(2.5);
  torch::Tensor result = torch::rsub(input, other, alpha);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::rsub(xla_input, other, alpha);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestConv2DBackward) {
  int in_channels = 4;
  int out_channels = 8;
  int kernel_size = 5;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        for (int dilation = 1; dilation <= 3; ++dilation) {
          for (int groups :
               {1, 2, 4}) {  // covers normal, grouped, depthwise conv.
            auto testfn =
                [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
              return torch::conv2d(inputs[0], inputs[1], inputs[2],
                                   /*stride=*/{stride, stride},
                                   /*padding=*/{padding, padding},
                                   /*dilation=*/{dilation, dilation}, groups);
            };

            ForEachDevice([&](const torch::Device& device) {
              torch::Tensor bias =
                  with_bias ? torch::rand({out_channels},
                                          torch::TensorOptions(torch::kDouble))
                            : torch::Tensor();
              TestBackward({torch::rand({1, in_channels, 14, 14},
                                        torch::TensorOptions(torch::kDouble)
                                            .requires_grad(true)),
                            torch::rand({out_channels, in_channels / groups,
                                         kernel_size, kernel_size},
                                        torch::TensorOptions(torch::kDouble)
                                            .requires_grad(true)),
                            bias},
                           device, testfn);
            });
          }
        };
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestTransposedConv2DBackward) {
  int in_channels = 4;
  int out_channels = 8;
  int kernel_size = 5;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (int dilation = 1; dilation <= 2; ++dilation) {
        for (int output_padding = 0;
             output_padding < std::max(stride, dilation); ++output_padding) {
          for (bool with_bias : {true, false}) {
            for (int groups :
                 {1, 2, 4}) {  // covers normal, grouped, depthwise conv.
              auto testfn = [&](const std::vector<torch::Tensor>& inputs)
                  -> torch::Tensor {
                return torch::conv_transpose2d(
                    inputs[0], inputs[1], inputs[2],
                    /*stride=*/{stride, stride + 1},
                    /*padding=*/{padding, padding + 1},
                    /*output_padding=*/output_padding,
                    /*groups=*/groups,
                    /*dilation=*/{dilation, dilation + 1});
              };
              ForEachDevice([&](const torch::Device& device) {
                torch::Tensor input = torch::rand(
                    {4, out_channels, 14, 14},
                    torch::TensorOptions(torch::kFloat).requires_grad(true));
                torch::Tensor weight = torch::rand(
                    {out_channels, in_channels / groups, kernel_size,
                     kernel_size},
                    torch::TensorOptions(torch::kFloat).requires_grad(true));
                torch::Tensor bias =
                    with_bias ? torch::rand({in_channels},
                                            torch::TensorOptions(torch::kFloat)
                                                .requires_grad(true))
                              : torch::Tensor();
                TestBackward({input, weight, bias}, device, testfn,
                             /*rtol=*/1e-5, /*atol=*/1e-5);
              });
            }
          };
        }
      }
    }
  }
}

}  // namespace cpp_test
}  // namespace torch_xla
