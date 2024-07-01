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

TEST_F(AtenXlaTensorTest, TestGettingSizeOnDynamicTensor) {
  // Make sure doing tensor.size() in c++ on dynamic tensor should fail.
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor b = torch::tensor({{0.0, 1.0}, {0.0, 0.0}},
                                    torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_nonzero = torch::nonzero(xla_b);
    EXPECT_THROW(xla_nonzero.sizes(), std::runtime_error);
    EXPECT_NO_THROW(xla_nonzero.sym_sizes());
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
  std::optional<std::string_view> rounding_modes[] = {"trunc", "floor",
                                                      std::nullopt};
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
  std::optional<std::string_view> rounding_modes[] = {"trunc", "floor",
                                                      std::nullopt};
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

TEST_F(AtenXlaTensorTest, TestDivScalarHalfOverflow) {
  torch::Tensor input = torch::rand({3, 4}, torch::TensorOptions(torch::kHalf));
  torch::Scalar other = torch::Scalar(100000);
  torch::Tensor out = torch::div(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_out = torch::div(xla_input, other);
    AllClose(out, xla_out);
  });
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

TEST_F(AtenXlaTensorTest, TestNllLoss2d) {
  int batch = 6;
  int classes = 2;
  int height = 3;
  int width = 3;
  for (auto dtype : {torch::kFloat, torch::kDouble}) {
    for (int ignore_index : {-1, 0, 1, 5}) {
      for (bool def_weight : {false, true}) {
        torch::Tensor input = torch::rand({batch, classes, height, width},
                                          torch::TensorOptions(dtype));
        torch::Tensor target = torch::randint(
            std::min(ignore_index, 0), classes, {batch, height, width},
            torch::TensorOptions(torch::kLong));
        torch::Tensor weight;
        if (def_weight) {
          weight = torch::rand({classes}, torch::TensorOptions(dtype));
        }
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean, torch::Reduction::Sum,
              torch::Reduction::None}) {
          torch::Tensor output =
              torch::nll_loss2d(/*self=*/input, /*target=*/target,
                                /*weight=*/weight,
                                /*reduction=*/reduction,
                                /*ignore_index=*/ignore_index);

          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_target = CopyToDevice(target, device);
            torch::Tensor xla_weight =
                def_weight ? CopyToDevice(weight, device) : torch::Tensor();
            torch::Tensor xla_output = torch::nll_loss2d(
                /*self=*/xla_input, /*target=*/xla_target,
                /*weight=*/xla_weight,
                /*reduction=*/reduction, /*ignore_index=*/ignore_index);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::nll_loss2d_forward",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSmoothL1Loss) {
  torch::Tensor input =
      torch::randn({2, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target =
      torch::randn({2, 4}, torch::TensorOptions(torch::kFloat));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None, torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    for (double beta : {0.25, 1.}) {
      torch::Tensor output =
          torch::smooth_l1_loss(input, target, reduction, beta);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_target = CopyToDevice(target, device);
        torch::Tensor xla_output =
            torch::smooth_l1_loss(xla_input, xla_target, reduction, beta);
        AllClose(output, xla_output);
      });
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::smooth_l1_loss", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestL1Loss) {
  torch::Tensor input =
      torch::randn({2, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target =
      torch::randn({2, 4}, torch::TensorOptions(torch::kFloat));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None, torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    torch::Tensor output = torch::l1_loss(input, target, reduction);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_target = CopyToDevice(target, device);
      torch::Tensor xla_output =
          torch::l1_loss(xla_input, xla_target, reduction);
      AllClose(output, xla_output);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestL1LossBackward) {
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None, torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::l1_loss(inputs[0], inputs[1], reduction);
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand({2, 4},
                       torch::TensorOptions(torch::kFloat).requires_grad(true)),
           torch::rand({2, 4}, torch::TensorOptions(torch::kFloat))},
          device, testfn);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMseLoss) {
  torch::Tensor input =
      torch::randn({2, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target =
      torch::randn({2, 4}, torch::TensorOptions(torch::kFloat));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None, torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    torch::Tensor output = torch::mse_loss(input, target, reduction);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_target = CopyToDevice(target, device);
      torch::Tensor xla_output =
          torch::mse_loss(xla_input, xla_target, reduction);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMseLossBackward) {
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None, torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::mse_loss(inputs[0], inputs[1], reduction);
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand({2, 4},
                       torch::TensorOptions(torch::kFloat).requires_grad(true)),
           torch::rand({2, 4}, torch::TensorOptions(torch::kFloat))},
          device, testfn);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestBatchNorm1D) {
  int num_features = 3;
  torch::Tensor input =
      torch::rand({2, num_features, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({num_features}, torch::TensorOptions(torch::kFloat));
  torch::Tensor bias =
      torch::rand({num_features}, torch::TensorOptions(torch::kFloat));
  torch::Tensor running_mean =
      torch::zeros({num_features}, torch::TensorOptions(torch::kFloat));
  torch::Tensor running_var =
      torch::ones({num_features}, torch::TensorOptions(torch::kFloat));
  double momentum = 0.1;
  double eps = 0.5;
  torch::Tensor undef;
  for (bool training : {true, false}) {
    for (bool undef_weight_bias : {false, true}) {
      torch::Tensor output = torch::batch_norm(
          /*input=*/input, /*weight=*/undef_weight_bias ? undef : weight,
          /*bias=*/undef_weight_bias ? undef : bias,
          /*running_mean=*/running_mean, /*running_var=*/running_var,
          /*training=*/training, /*momentum=*/momentum, /*eps=*/eps,
          /*cudnn_enabled=*/false);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_weight =
            undef_weight_bias ? undef : CopyToDevice(weight, device);
        torch::Tensor xla_bias =
            undef_weight_bias ? undef : CopyToDevice(bias, device);
        torch::Tensor xla_running_mean = CopyToDevice(running_mean, device);
        torch::Tensor xla_running_var = CopyToDevice(running_var, device);
        torch::Tensor xla_output = torch::batch_norm(
            /*input=*/xla_input, /*weight=*/xla_weight, /*bias=*/xla_bias,
            /*running_mean=*/xla_running_mean, /*running_var=*/xla_running_var,
            /*training=*/training, /*momentum=*/momentum, /*eps=*/eps,
            /*cudnn_enabled=*/false);
        AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::native_batch_norm",
                           cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestBatchNorm2D) {
  int num_features = 3;
  torch::Tensor input =
      torch::rand({2, num_features, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({num_features}, torch::TensorOptions(torch::kFloat));
  torch::Tensor bias =
      torch::rand({num_features}, torch::TensorOptions(torch::kFloat));
  torch::Tensor running_mean =
      torch::zeros({num_features}, torch::TensorOptions(torch::kFloat));
  torch::Tensor running_var =
      torch::ones({num_features}, torch::TensorOptions(torch::kFloat));
  double momentum = 0.1;
  double eps = 0.5;
  torch::Tensor undef;
  for (bool training : {true, false}) {
    for (bool undef_weight_bias : {false, true}) {
      torch::Tensor output = torch::batch_norm(
          /*input=*/input, /*weight=*/undef_weight_bias ? undef : weight,
          /*bias=*/undef_weight_bias ? undef : bias,
          /*running_mean=*/running_mean, /*running_var=*/running_var,
          /*training=*/training, /*momentum=*/momentum, /*eps=*/eps,
          /*cudnn_enabled=*/false);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_weight =
            undef_weight_bias ? undef : CopyToDevice(weight, device);
        torch::Tensor xla_bias =
            undef_weight_bias ? undef : CopyToDevice(bias, device);
        torch::Tensor xla_running_mean = CopyToDevice(running_mean, device);
        torch::Tensor xla_running_var = CopyToDevice(running_var, device);
        torch::Tensor xla_output = torch::batch_norm(
            /*input=*/xla_input, /*weight=*/xla_weight, /*bias=*/xla_bias,
            /*running_mean=*/xla_running_mean, /*running_var=*/xla_running_var,
            /*training=*/training, /*momentum=*/momentum, /*eps=*/eps,
            /*cudnn_enabled=*/false);
        AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::native_batch_norm",
                           cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestDim) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    EXPECT_EQ(input.dim(), xla_input.dim());
  });
}

TEST_F(AtenXlaTensorTest, TestContiguous) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::native::contiguous(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::native::contiguous(xla_input);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_copy_from", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSqueezeAll) {
  torch::Tensor input =
      torch::rand({2, 1, 3, 1}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::squeeze(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::squeeze(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestSqueezeAllInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor input =
        torch::rand({2, 1, 3, 1}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = input.squeeze_();
    torch::Tensor xla_output = xla_input.squeeze_();
    AllClose(output, xla_output);
    AllClose(input, xla_input);
    ASSERT_EQ(input.dim(), xla_input.dim());
    for (int64_t dim_idx = 0; dim_idx < input.dim(); ++dim_idx) {
      ASSERT_EQ(input.size(dim_idx), xla_input.size(dim_idx));
    }
  });
}

TEST_F(AtenXlaTensorTest, TestSqueezeOne) {
  torch::Tensor input =
      torch::rand({2, 1, 3, 1}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor output = torch::squeeze(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::squeeze(xla_input, dim);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestSqueezeMultipleDims) {
  torch::Tensor input =
      torch::rand({2, 1, 3, 1}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> dims = {1, 2, 3};
  torch::Tensor output = torch::squeeze(input, dims);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::squeeze(xla_input, dims);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestSqueezeDimWithNegativeOne) {
  torch::Tensor input =
      torch::rand({2, 1, 3, 1}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> dims = {-1};
  torch::Tensor output = torch::squeeze(input, dims);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::squeeze(xla_input, dims);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestSqueezeOneInPlace) {
  int rank = 4;
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor input =
          torch::rand({2, 1, 3, 1}, torch::TensorOptions(torch::kFloat));
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor output = input.squeeze_(dim);
      torch::Tensor xla_output = xla_input.squeeze_(dim);
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
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim() + 1;
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor output = torch::unsqueeze(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::unsqueeze(xla_input, dim);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestUnsqueezeInPlace) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim() + 1;
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor output = input.unsqueeze_(dim);
      torch::Tensor xla_output = xla_input.unsqueeze_(dim);
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
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor mask =
      torch::randint(0, 2, {2, 3}, torch::TensorOptions(torch::kBool));
  torch::Scalar value(42);
  torch::Tensor result = torch::masked_fill(input, mask, value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_mask = CopyToDevice(mask, device);
    torch::Tensor xla_result = torch::masked_fill(xla_input, xla_mask, value);
    AllClose(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::masked_fill", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMaskedFillInPlace) {
  torch::Scalar value(42);
  torch::Tensor mask =
      torch::randint(0, 2, {2, 3}, torch::TensorOptions(torch::kBool));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor input =
        torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_mask = CopyToDevice(mask, device);
    torch::Tensor result = input.masked_fill_(mask, value);
    torch::Tensor xla_result = xla_input.masked_fill_(xla_mask, value);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::masked_fill", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMaskedFillBroadcast1) {
  torch::Tensor input =
      torch::rand({2, 5, 4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor mask =
      torch::randint(0, 2, {4, 1}, torch::TensorOptions(torch::kBool));
  torch::Scalar value(42);
  torch::Tensor result = torch::masked_fill(input, mask, value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_mask = CopyToDevice(mask, device);
    torch::Tensor xla_result = torch::masked_fill(xla_input, xla_mask, value);
    AllClose(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::masked_fill", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMaskedFillBroadcast2) {
  torch::Tensor input =
      torch::rand({2, 1}, torch::TensorOptions(torch::kFloat));
  torch::Tensor mask =
      torch::randint(0, 2, {2, 3}, torch::TensorOptions(torch::kBool));
  torch::Scalar value(42);
  torch::Tensor result = torch::masked_fill(input, mask, value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_mask = CopyToDevice(mask, device);
    torch::Tensor xla_result = torch::masked_fill(xla_input, xla_mask, value);
    AllClose(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::masked_fill", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMaskedFillBroadcast3) {
  torch::Tensor input =
      torch::rand({2, 1}, torch::TensorOptions(torch::kFloat));
  torch::Tensor mask =
      torch::randint(0, 2, {4, 2, 3}, torch::TensorOptions(torch::kBool));
  torch::Scalar value(42);
  torch::Tensor result = torch::masked_fill(input, mask, value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_mask = CopyToDevice(mask, device);
    torch::Tensor xla_result = torch::masked_fill(xla_input, xla_mask, value);
    AllClose(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::masked_fill", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestFill) {
  torch::Scalar value(42);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor input =
        torch::empty({2, 3}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor result = torch::fill_(input, value);
    torch::Tensor xla_result = torch::fill_(xla_input, value);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestFillWithRank0) {
  torch::Tensor value = torch::scalar_tensor(42);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor input =
        torch::empty({2, 3}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor result = torch::fill_(input, value);
    torch::Tensor xla_value = CopyToDevice(value, device);
    torch::Tensor xla_result = torch::fill_(xla_input, value);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestPermute) {
  torch::Tensor input =
      torch::rand({2, 3, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<std::vector<int64_t>> dims_permutations = {
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  int rank = input.dim();
  for (std::vector<int64_t> dims_permutation : dims_permutations) {
    for (bool negative_dims : {false, true}) {
      if (negative_dims) {
        std::for_each(dims_permutation.begin(), dims_permutation.end(),
                      [rank](int64_t& dim) { dim -= rank; });
      }
      torch::Tensor output = input.permute(dims_permutation);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_output = xla_input.permute(dims_permutation);
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
      torch::Tensor input =
          torch::zeros(input_sizes, torch::TensorOptions(torch::kFloat));
      torch::Tensor one =
          torch::tensor(1.0, torch::TensorOptions(torch::kFloat));
      torch::Tensor output = input.permute(dims_permutation);
      output.add_(one, 1.0);
      input.add_(one, 1.0);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xinput =
            torch::zeros(input_sizes, torch::TensorOptions(torch::kFloat));
        torch::Tensor xla_input = CopyToDevice(xinput, device);
        torch::Tensor xla_one = CopyToDevice(one, device);
        torch::Tensor xla_output = xla_input.permute(dims_permutation);
        xla_output.add_(xla_one, 1.0);
        xla_input.add_(xla_one, 1.0);
        AllClose(output, xla_output);
        AllClose(input, xla_input);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestFlip) {
  torch::Tensor input =
      torch::rand({2, 3, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<std::vector<int64_t>> dim_powerset = {
      {0}, {1}, {2}, {0, 1}, {1, 2}, {2, 0}, {0, 1, 2}};
  for (std::vector<int64_t> flip_dims : dim_powerset) {
    for (bool negative_dims : {false, true}) {
      if (negative_dims) {
        std::for_each(flip_dims.begin(), flip_dims.end(),
                      [](int64_t& dim) { dim -= 3; });
      }
      torch::Tensor output = torch::flip(input, flip_dims);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_output = torch::flip(xla_input, flip_dims);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestPixelShuffle) {
  torch::Tensor input =
      torch::rand({5, 18, 4, 4}, torch::TensorOptions(torch::kFloat));
  int upscale_factor = 3;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::pixel_shuffle(input, upscale_factor);
    torch::Tensor xla_output = torch::pixel_shuffle(xla_input, upscale_factor);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSumToSize) {
  torch::Tensor input =
      torch::rand({4, 6, 3, 7}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> out_size = {4, 1, 1, 7};
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = input.sum_to_size(out_size);
    torch::Tensor xla_output = xla_input.sum_to_size(out_size);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::sum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestTransposeDims) {
  torch::Tensor input =
      torch::rand({2, 3, 4}, torch::TensorOptions(torch::kFloat));
  int dim0 = 0;
  int dim1 = 2;
  torch::Tensor output = torch::transpose(input, dim0, dim1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::transpose(xla_input, dim0, dim1);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestTransposeDimsMod) {
  std::vector<int64_t> input_sizes = {2, 3, 4};
  int dim0 = 0;
  int dim1 = 2;
  torch::Tensor input =
      torch::zeros(input_sizes, torch::TensorOptions(torch::kFloat));
  torch::Tensor one = torch::tensor(1.0, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::transpose(input, dim0, dim1);
  output.add_(one, 1.0);
  input.add_(one, 1.0);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xinput =
        torch::zeros(input_sizes, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(xinput, device);
    torch::Tensor xla_one = CopyToDevice(one, device);
    torch::Tensor xla_output = torch::transpose(xla_input, dim0, dim1);
    xla_output.add_(xla_one, 1.0);
    xla_input.add_(xla_one, 1.0);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestTransposeDimsInPlace) {
  torch::Tensor input =
      torch::rand({2, 3, 4}, torch::TensorOptions(torch::kFloat));
  int dim0 = 0;
  int dim1 = 2;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = input.transpose_(dim0, dim1);
    torch::Tensor xla_output = xla_input.transpose_(dim0, dim1);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestSplit) {
  torch::Tensor input =
      torch::rand({7, 8, 9}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int split_size : {2, 3}) {
    for (int dim = -rank; dim < rank; ++dim) {
      std::vector<torch::Tensor> outputs = torch::split(input, split_size, dim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        std::vector<torch::Tensor> xla_outputs =
            torch::split(xla_input, split_size, dim);
        ASSERT_EQ(outputs.size(), xla_outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
          AllClose(outputs[i], xla_outputs[i]);
        }
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestSplitEmpty) {
  torch::Tensor input = torch::rand({0}, torch::TensorOptions(torch::kFloat));
  int split_size = 0;
  int dim = 0;
  std::vector<torch::Tensor> outputs = torch::split(input, split_size, dim);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    std::vector<torch::Tensor> xla_outputs =
        torch::split(xla_input, split_size, dim);
    ASSERT_EQ(outputs.size(), xla_outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      AllClose(outputs[i], xla_outputs[i]);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestSplitWithSizes) {
  torch::Tensor input =
      torch::rand({15, 15, 15}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    std::vector<torch::Tensor> outputs =
        torch::split_with_sizes(input, {4, 5, 6}, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      std::vector<torch::Tensor> xla_outputs =
          torch::split_with_sizes(xla_input, {4, 5, 6}, dim);
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
    torch::Tensor input =
        torch::rand(dim_size, torch::TensorOptions(torch::kFloat));
    torch::Tensor other =
        torch::rand(dim_size, torch::TensorOptions(torch::kFloat));
    torch::Tensor result = torch::cross(input, other);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_other = CopyToDevice(other, device);
      torch::Tensor xla_result = torch::cross(xla_input, xla_other);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCrossExplicitDim) {
  std::vector<int64_t> dim_size = {3, 3};
  torch::Tensor input =
      torch::rand(dim_size, torch::TensorOptions(torch::kFloat));
  torch::Tensor other =
      torch::rand(dim_size, torch::TensorOptions(torch::kFloat));
  int rank = dim_size.size();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cross(input, other, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_other = CopyToDevice(other, device);
      torch::Tensor xla_result = torch::cross(xla_input, xla_other, dim);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCrossZeroDim) {
  torch::Tensor input =
      torch::rand({0, 1, 3, 0}, torch::TensorOptions(torch::kFloat));
  torch::Tensor result = torch::cross(input, input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::cross(xla_input, xla_input);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestTriu) {
  int size = 5;
  torch::Tensor input =
      torch::rand({size, size}, torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::triu(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::triu(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::triu", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestTriuNonSquare) {
  int size = 5;
  torch::Tensor input =
      torch::rand({size, size + 1}, torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::triu(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::triu(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::triu", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestTriuBatch) {
  int size = 5;
  int batch_size = 3;
  torch::Tensor input = torch::rand({batch_size, size, size},
                                    torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::triu(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::triu(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::triu", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestTril) {
  int size = 5;
  torch::Tensor input =
      torch::rand({size, size}, torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::tril(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::tril(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::tril", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestTrilNonSquare) {
  int size = 5;
  torch::Tensor input =
      torch::rand({size, size + 1}, torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::tril(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::tril(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::tril", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestTrilBatch) {
  int size = 5;
  int batch_size = 3;
  torch::Tensor input = torch::rand({batch_size, size, size},
                                    torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::tril(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::tril(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::tril", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestTriuInPlace) {
  int size = 5;
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor input =
          torch::rand({size, size}, torch::TensorOptions(torch::kFloat));
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor output = input.triu_(diagonal);
      torch::Tensor xla_output = xla_input.triu_(diagonal);
      AllClose(output, xla_output);
      AllClose(input, xla_input);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::triu", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestTrilInPlace) {
  int size = 5;
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor input =
          torch::rand({size, size}, torch::TensorOptions(torch::kFloat));
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor output = input.tril_(diagonal);
      torch::Tensor xla_output = xla_input.tril_(diagonal);
      AllClose(output, xla_output);
      AllClose(input, xla_input);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::tril", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestTrace) {
  int n = 5;
  torch::Tensor input =
      torch::rand({n, n}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::trace(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::trace(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestTraceWide) {
  int lines = 3;
  int cols = 5;
  torch::Tensor input =
      torch::rand({lines, cols}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::trace(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::trace(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestTraceNarrow) {
  int lines = 5;
  int cols = 3;
  torch::Tensor input =
      torch::rand({lines, cols}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::trace(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::trace(xla_input);
    AllClose(output, xla_output);
  });
}

}  // namespace cpp_test
}  // namespace torch_xla
