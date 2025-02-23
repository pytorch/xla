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

TEST_F(AtenXlaTensorTest, TestDiagRank1) {
  int size = 7;
  torch::Tensor input =
      torch::rand({size}, torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -2 * size; diagonal <= 2 * size; ++diagonal) {
    torch::Tensor output = torch::diag(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::diag(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestDiagRank2) {
  int size = 7;
  torch::Tensor input =
      torch::rand({size, size}, torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::diag(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::diag(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestDiagFlat) {
  torch::Tensor input =
      torch::rand({4, 3, 6, 7}, torch::TensorOptions(torch::kFloat));
  for (int diagonal = -10; diagonal < 10; ++diagonal) {
    torch::Tensor output = torch::diagflat(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::diagflat(xla_input, diagonal);
      AllClose(output, xla_output);
    });

    ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::zero_", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::view_copy_symint",
                         cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::_to_copy", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::_copy_from", cpp_test::GetIgnoredCounters());
  }
}

TEST_F(AtenXlaTensorTest, TestDiagonal) {
  int size = 5;
  torch::Tensor input =
      torch::rand({size, size}, torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::diagonal(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::diagonal(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestDiagonalNonSquare) {
  int size = 5;
  torch::Tensor input =
      torch::rand({size, size + 1}, torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::diagonal(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::diagonal(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestDiagonalBatch) {
  int size = 5;
  int batch_size = 3;
  int dim1 = 1;
  int dim2 = 2;
  torch::Tensor input = torch::rand({batch_size, size, size},
                                    torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output =
        torch::diagonal(input, diagonal, /*dim1=*/dim1, /*dim1=*/dim2);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output =
          torch::diagonal(xla_input, diagonal, /*dim1=*/dim1, /*dim1=*/dim2);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestFlatten) {
  torch::Tensor input = torch::rand({4, 7, 5, 3});
  int rank = input.dim();
  for (int pos_start_dim = 0; pos_start_dim < rank; ++pos_start_dim) {
    for (int pos_end_dim = pos_start_dim; pos_end_dim < rank; ++pos_end_dim) {
      for (bool negative_start_dim : {false, true}) {
        for (bool negative_end_dim : {false, true}) {
          int start_dim =
              negative_start_dim ? pos_start_dim - rank : pos_start_dim;
          int end_dim = negative_end_dim ? pos_end_dim - rank : pos_end_dim;
          torch::Tensor output = torch::flatten(input, start_dim, end_dim);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output =
                torch::flatten(xla_input, start_dim, end_dim);
            AllClose(output, xla_output);
          });

          ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
          // Depends on shapes, flatten could call into different view
          // functions. So we skip positive checks here.
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestLogicalNot) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor input =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type))
            : torch::randint(0, 100, {3, 4}, torch::TensorOptions(scalar_type));
    torch::Tensor result = torch::logical_not(input);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::logical_not(xla_input);
      AllEqual(result, xla_result);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::logical_not", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLogicalXor) {
  for (torch::ScalarType scalar_type1 :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor lhs =
        isFloatingType(scalar_type1)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
            : torch::randint(0, 100, {3, 4},
                             torch::TensorOptions(scalar_type1));
    for (torch::ScalarType scalar_type2 :
         {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
          torch::kLong}) {
      torch::Tensor rhs =
          isFloatingType(scalar_type2)
              ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
              : torch::randint(1, 100, {3, 4},
                               torch::TensorOptions(scalar_type2));
      torch::Tensor result = torch::logical_xor(lhs, rhs);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_lhs = CopyToDevice(lhs, device);
        torch::Tensor xla_rhs = CopyToDevice(rhs, device);
        torch::Tensor xla_result = torch::logical_xor(xla_lhs, xla_rhs);
        AllEqual(result, xla_result);
      });
    }
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::logical_xor", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLogicalAnd) {
  for (torch::ScalarType scalar_type1 :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor lhs =
        isFloatingType(scalar_type1)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
            : torch::randint(0, 100, {3, 4},
                             torch::TensorOptions(scalar_type1));
    for (torch::ScalarType scalar_type2 :
         {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
          torch::kLong}) {
      torch::Tensor rhs =
          isFloatingType(scalar_type2)
              ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
              : torch::randint(1, 100, {3, 4},
                               torch::TensorOptions(scalar_type2));
      torch::Tensor result = torch::logical_and(lhs, rhs);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_lhs = CopyToDevice(lhs, device);
        torch::Tensor xla_rhs = CopyToDevice(rhs, device);
        torch::Tensor xla_result = torch::logical_and(xla_lhs, xla_rhs);
        AllEqual(result, xla_result);
      });
    }
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::logical_and", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLogicalOr) {
  for (torch::ScalarType scalar_type1 :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor lhs =
        isFloatingType(scalar_type1)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
            : torch::randint(0, 100, {3, 4},
                             torch::TensorOptions(scalar_type1));
    for (torch::ScalarType scalar_type2 :
         {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
          torch::kLong}) {
      torch::Tensor rhs =
          isFloatingType(scalar_type2)
              ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
              : torch::randint(1, 100, {3, 4},
                               torch::TensorOptions(scalar_type2));
      torch::Tensor result = torch::logical_or(lhs, rhs);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_lhs = CopyToDevice(lhs, device);
        torch::Tensor xla_rhs = CopyToDevice(rhs, device);
        torch::Tensor xla_result = torch::logical_or(xla_lhs, xla_rhs);
        AllEqual(result, xla_result);
      });
    }
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::logical_or", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBitwiseAnd) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor result = lhs.__and__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor xla_rhs = CopyToDevice(rhs, device);
    torch::Tensor xla_result = xla_lhs.__and__(xla_rhs);
    AllEqual(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bitwise_and", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBitwiseAndInPlace) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__iand__(rhs);
    torch::Tensor xla_rhs = CopyToDevice(rhs, device);
    torch::Tensor xla_result = xla_lhs.__iand__(xla_rhs);
    AllEqual(result, xla_result);
    AllEqual(lhs, xla_lhs);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bitwise_and", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBitwiseAndScalar) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  torch::Tensor result = lhs.__and__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor xla_result = xla_lhs.__and__(rhs);
    AllEqual(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bitwise_and", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBitwiseAndScalarInPlace) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__iand__(rhs);
    torch::Tensor xla_result = xla_lhs.__iand__(rhs);
    AllEqual(result, xla_result);
    AllEqual(lhs, xla_lhs);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bitwise_and", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBitwiseAndPromotion) {
  torch::Tensor input =
      torch::rand({4, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor view = input.reshape(-1);
  torch::Tensor result = torch::__and__(view.gt(0), view.ne(0));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_view = xla_input.reshape(-1);
    torch::Tensor xla_result = torch::__and__(xla_view.gt(0), xla_view.ne(0));
    AllEqual(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bitwise_and", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBitwiseOr) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor result = lhs.__or__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor xla_rhs = CopyToDevice(rhs, device);
    torch::Tensor xla_result = xla_lhs.__or__(xla_rhs);
    AllEqual(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bitwise_or", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBitwiseOrInPlace) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__ior__(rhs);
    torch::Tensor xla_rhs = CopyToDevice(rhs, device);
    torch::Tensor xla_result = xla_lhs.__ior__(xla_rhs);
    AllEqual(result, xla_result);
    AllEqual(lhs, xla_lhs);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bitwise_or", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBitwiseOrScalar) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  torch::Tensor result = lhs.__or__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor xla_result = xla_lhs.__or__(rhs);
    AllEqual(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bitwise_or", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBitwiseOrScalarInPlace) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__ior__(rhs);
    torch::Tensor xla_result = xla_lhs.__ior__(rhs);
    AllEqual(result, xla_result);
    AllEqual(lhs, xla_lhs);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bitwise_or", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBitwiseXor) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor result = lhs.__xor__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor xla_rhs = CopyToDevice(rhs, device);
    torch::Tensor xla_result = xla_lhs.__xor__(xla_rhs);
    AllEqual(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bitwise_xor", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBitwiseXorInPlace) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__ixor__(rhs);
    torch::Tensor xla_rhs = CopyToDevice(rhs, device);
    torch::Tensor xla_result = xla_lhs.__ixor__(xla_rhs);
    AllEqual(result, xla_result);
    AllEqual(lhs, xla_lhs);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bitwise_xor", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBitwiseXorScalar) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  torch::Tensor result = lhs.__xor__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor xla_result = xla_lhs.__xor__(rhs);
    AllEqual(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bitwise_xor", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBitwiseXorScalarInPlace) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__ixor__(rhs);
    torch::Tensor xla_result = xla_lhs.__ixor__(rhs);
    AllEqual(result, xla_result);
    AllEqual(lhs, xla_lhs);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bitwise_xor", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLshift) {
  torch::Tensor input =
      torch::ones({4, 2}, torch::TensorOptions(torch::kInt32));
  torch::Tensor shift_amount =
      torch::randint(16, input.sizes(), torch::TensorOptions(torch::kInt32));
  torch::Tensor result = torch::__lshift__(input, shift_amount);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_shift_amount = CopyToDevice(shift_amount, device);
    torch::Tensor xla_result = torch::__lshift__(xla_input, xla_shift_amount);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestLshiftInPlace) {
  torch::Tensor input =
      torch::ones({4, 2}, torch::TensorOptions(torch::kInt32));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor shift_amount =
        torch::randint(16, input.sizes(), torch::TensorOptions(torch::kInt32));
    torch::Tensor result = input.__ilshift__(shift_amount);
    torch::Tensor xla_shift_amount = CopyToDevice(shift_amount, device);
    torch::Tensor xla_result = xla_input.__ilshift__(xla_shift_amount);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestLshiftScalar) {
  torch::Tensor input =
      torch::ones({4, 2}, torch::TensorOptions(torch::kInt32));
  torch::Scalar shift_amount = 3;
  torch::Tensor result = torch::__lshift__(input, shift_amount);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::__lshift__(xla_input, shift_amount);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestLshiftScalarInPlace) {
  torch::Tensor input =
      torch::ones({4, 2}, torch::TensorOptions(torch::kInt32));
  torch::Scalar shift_amount = 3;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor result = input.__ilshift__(shift_amount);
    torch::Tensor xla_result = xla_input.__ilshift__(shift_amount);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestRshift) {
  torch::Tensor input =
      torch::ones({4, 2}, torch::TensorOptions(torch::kInt32));
  torch::Tensor shift_amount =
      torch::randint(16, input.sizes(), torch::TensorOptions(torch::kInt32));
  torch::Tensor result = torch::__rshift__(input, shift_amount);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_shift_amount = CopyToDevice(shift_amount, device);
    torch::Tensor xla_result = torch::__rshift__(xla_input, xla_shift_amount);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestRshiftInPlace) {
  torch::Tensor input =
      torch::ones({4, 2}, torch::TensorOptions(torch::kInt32));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor shift_amount =
        torch::randint(16, input.sizes(), torch::TensorOptions(torch::kInt32));
    torch::Tensor result = input.__irshift__(shift_amount);
    torch::Tensor xla_shift_amount = CopyToDevice(shift_amount, device);
    torch::Tensor xla_result = xla_input.__irshift__(xla_shift_amount);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestRshiftScalar) {
  torch::Tensor input =
      torch::ones({4, 2}, torch::TensorOptions(torch::kInt32));
  torch::Scalar shift_amount = 3;
  torch::Tensor result = torch::__rshift__(input, shift_amount);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::__rshift__(xla_input, shift_amount);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestRshiftScalarInPlace) {
  torch::Tensor input =
      torch::ones({4, 2}, torch::TensorOptions(torch::kInt32));
  torch::Scalar shift_amount = 3;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor result = input.__irshift__(shift_amount);
    torch::Tensor xla_result = xla_input.__irshift__(shift_amount);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestMeshgrid) {
  torch::Tensor a = torch::rand({3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  auto d = torch::meshgrid({a, b, c});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    auto xla_d = torch::meshgrid({xla_a, xla_b, xla_c});
    EXPECT_EQ(d.size(), xla_d.size());
    for (size_t i = 0; i < d.size(); ++i) {
      AllClose(d[i], xla_d[i]);
    }
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::view_copy", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestConstantPad) {
  torch::Tensor input =
      torch::rand({4, 2, 5}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{1, 2, 3, 4, 5, 6};
  float pad_value = 5;
  torch::Tensor output = torch::constant_pad_nd(input, pad, pad_value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output =
        torch::constant_pad_nd(xla_input, pad, pad_value);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestConstantPadIncomplete) {
  torch::Tensor input =
      torch::rand({4, 2, 5}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{1, 2};
  float pad_value = 5;
  torch::Tensor output = torch::constant_pad_nd(input, pad, pad_value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output =
        torch::constant_pad_nd(xla_input, pad, pad_value);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestReflectionPad1dRank2) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{2, 2};
  torch::Tensor output = torch::reflection_pad1d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::reflection_pad1d(xla_input, pad);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::reflection_pad1d", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReflectionPad1dRank3) {
  torch::Tensor input =
      torch::rand({2, 3, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{2, 2};
  torch::Tensor output = torch::reflection_pad1d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::reflection_pad1d(xla_input, pad);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::reflection_pad1d", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReflectionPad1dBackward) {
  std::vector<int64_t> pad{2, 2};
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::reflection_pad1d(inputs[0], pad);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 2, 3},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReflectionPad2dRank3) {
  torch::Tensor input =
      torch::rand({2, 3, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{2, 2, 2, 2};
  torch::Tensor output = torch::reflection_pad2d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::reflection_pad2d(xla_input, pad);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::reflection_pad2d", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReflectionPad2dRank4) {
  torch::Tensor input =
      torch::rand({2, 2, 3, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{2, 2, 2, 2};
  torch::Tensor output = torch::reflection_pad2d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::reflection_pad2d(xla_input, pad);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::reflection_pad2d", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReflectionPad2dBackward) {
  std::vector<int64_t> pad{2, 3, 1, 2};
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::reflection_pad2d(inputs[0], pad);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({1, 2, 4, 4},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReflectionPad3dRank5) {
  torch::Tensor input =
      torch::rand({2, 2, 3, 4, 2}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{1, 1, 1, 2, 2, 1};
  torch::Tensor output = torch::reflection_pad3d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::reflection_pad3d(xla_input, pad);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::reflection_pad3d", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReflectionPad3dRank4) {
  torch::Tensor input =
      torch::rand({2, 2, 3, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{1, 1, 1, 1, 1, 1};
  torch::Tensor output = torch::reflection_pad3d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::reflection_pad3d(xla_input, pad);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::reflection_pad3d", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReflectionPad3dBackward) {
  std::vector<int64_t> pad{1, 1, 1, 1, 1, 1};
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::reflection_pad3d(inputs[0], pad);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 2, 4, 4, 2},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReplicationPad1d) {
  torch::Tensor input =
      torch::rand({1, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{1, 2};
  torch::Tensor output = torch::replication_pad1d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::replication_pad1d(xla_input, pad);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::replication_pad1d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReplicationPad1dZeroPad) {
  torch::Tensor input =
      torch::rand({1, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{1, 0};
  torch::Tensor output = torch::replication_pad1d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::replication_pad1d(xla_input, pad);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::replication_pad1d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReplicationPad1dBackward) {
  std::vector<int64_t> pad{2, 3};
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::replication_pad1d(inputs[0], pad);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 4},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReplicationPad2d) {
  torch::Tensor input =
      torch::rand({1, 3, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{1, 2, 2, 1};
  torch::Tensor output = torch::replication_pad2d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::replication_pad2d(xla_input, pad);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::replication_pad2d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReplicationPad2dZeroPad) {
  torch::Tensor input =
      torch::rand({1, 3, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{1, 0, 0, 1};
  torch::Tensor output = torch::replication_pad2d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::replication_pad2d(xla_input, pad);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::replication_pad2d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReplicationPad2dBackward) {
  std::vector<int64_t> pad{2, 3, 1, 1};
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::replication_pad2d(inputs[0], pad);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 3, 4},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReplicationPad3d) {
  torch::Tensor input =
      torch::rand({1, 3, 4, 5}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{1, 2, 2, 2, 2, 1};
  torch::Tensor output = torch::replication_pad3d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::replication_pad3d(xla_input, pad);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::replication_pad3d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReplicationPad3dZeroPad) {
  torch::Tensor input =
      torch::rand({1, 3, 4, 5}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{1, 0, 0, 0, 0, 1};
  torch::Tensor output = torch::replication_pad3d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::replication_pad3d(xla_input, pad);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::replication_pad3d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestReplicationPad3dBackward) {
  std::vector<int64_t> pad{2, 3, 1, 1, 1, 1};
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::replication_pad3d(inputs[0], pad);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 3, 4, 5},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAsStrided) {
  torch::Tensor input =
      torch::rand({128, 320}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> size = {128, 20, 4, 4};
  std::vector<int64_t> stride = {320, 16, 4, 1};
  torch::Tensor output =
      torch::as_strided(input, /*size=*/size, /*stride=*/stride);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output =
        torch::as_strided(xla_input, /*size=*/size, /*stride=*/stride);
    AllClose(output, xla_output);
  });
  ExpectCounterNotChanged("aten::*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("xla::take", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::as_strided_copy", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAsStridedInPlace) {
  torch::Tensor input =
      torch::rand({128, 320}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> size = {128, 20, 4, 4};
  std::vector<int64_t> stride = {320, 16, 4, 1};
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output =
        torch::as_strided_(input, /*size=*/size, /*stride=*/stride);
    torch::Tensor xla_output =
        torch::as_strided_(xla_input, /*size=*/size, /*stride=*/stride);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
  ExpectCounterNotChanged("aten::*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("xla::take", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::as_strided_copy", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAsStridedWithOffset) {
  torch::Tensor input =
      torch::rand({4, 8, 2}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> size = {4, 4, 2};
  std::vector<int64_t> stride = {8, 2, 1};
  int64_t storage_offset = 4;
  torch::Tensor output =
      torch::as_strided(input, /*size=*/size, /*stride=*/stride,
                        /*storage_offset=*/storage_offset);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output =
        torch::as_strided(xla_input, /*size=*/size, /*stride=*/stride,
                          /*storage_offset=*/storage_offset);
    AllClose(output, xla_output);
  });
  ExpectCounterNotChanged("aten::*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("xla::take", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::as_strided_copy", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAsStridedWithInplaceCopy) {
  torch::Tensor grad = torch::ones({4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> size = {4};
  std::vector<int64_t> stride = {1};
  torch::Tensor output = torch::zeros({4}, grad.options());
  output.as_strided(size, stride).copy_(grad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_grad = CopyToDevice(grad, device);
    torch::Tensor xla_output = torch::zeros({4}, xla_grad.options());
    xla_output.as_strided(size, stride).copy_(xla_grad);
    AllClose(output, xla_output);
  });
  ExpectCounterNotChanged("aten::*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("xla::take", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::as_strided_copy", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEmptyStrided) {
  std::vector<int64_t> size = {4, 4, 2};
  std::vector<int64_t> stride = {8, 2, 1};
  torch::Tensor output = torch::empty_strided(/*size=*/size, /*stride=*/stride);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_output =
        torch::empty_strided(/*size=*/size, /*stride=*/stride);
    EXPECT_EQ(output.sizes(), xla_output.sizes());
    EXPECT_EQ(output.strides(), xla_output.strides());
  });
}

TEST_F(AtenXlaTensorTest, TestAsStridedUseSlice) {
  torch::Tensor input =
      torch::rand({16, 32, 24}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> size = {16, 8, 24};
  std::vector<int64_t> stride = {768, 48, 1};
  torch::Tensor output =
      torch::as_strided(input, /*size=*/size, /*stride=*/stride);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output =
        torch::as_strided(xla_input, /*size=*/size, /*stride=*/stride);
    AllClose(output, xla_output);
  });
  ExpectCounterNotChanged("aten::*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("xla::take", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::as_strided_copy", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAsStridedMismatchLastDimUseSlice) {
  torch::Tensor input =
      torch::rand({16, 32, 24}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> size = {16, 32}; // 16, 32, 24
  std::vector<int64_t> stride = {768, 24}; // 768, 24, 1
  torch::Tensor output =
      torch::as_strided(input, /*size=*/size, /*stride=*/stride);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output =
        torch::as_strided(xla_input, /*size=*/size, /*stride=*/stride);
    AllClose(output, xla_output);
  });
  ExpectCounterNotChanged("aten::*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("xla::take", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::as_strided_copy", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAsStridedMismatchMiddleDimUseSlice) {
  torch::lazy::MetricsArena::Get()->ResetMetrics();
  runtime::metrics::ClearMetrics();
  torch::Tensor input =
      torch::rand({6, 4, 2, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> size = {6, 2, 4};
  std::vector<int64_t> stride = {32, 4, 1};
  torch::Tensor output =
      torch::as_strided(input, /*size=*/size, /*stride=*/stride);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output =
        torch::as_strided(xla_input, /*size=*/size, /*stride=*/stride);
    AllClose(output, xla_output);
  });
  ExpectCounterNotChanged("aten::*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("xla::take", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::as_strided_copy", cpp_test::GetIgnoredCounters());
}



TEST_F(AtenXlaTensorTest, TestAsStridedMismatchDimWithOffset) {
  torch::lazy::MetricsArena::Get()->ResetMetrics();
  runtime::metrics::ClearMetrics();
  torch::Tensor input =
      torch::rand({6, 4, 2, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> size = {6, 2, 4};
  std::vector<int64_t> stride = {32, 4, 1};
  torch::Tensor output =
      torch::as_strided(input, /*size=*/size, /*stride=*/stride, 1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output =
        torch::as_strided(xla_input, /*size=*/size, /*stride=*/stride, 1);
    AllClose(output, xla_output);
  });
  ExpectCounterNotChanged("aten::*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::take", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::as_strided_copy", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAsStridedMultipleMismatchDimWithOffset) {
  torch::lazy::MetricsArena::Get()->ResetMetrics();
  runtime::metrics::ClearMetrics();
  torch::Tensor input =
      torch::rand({6, 4, 2, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> size = {3, 2, 4};
  std::vector<int64_t> stride = {16, 4, 1};
  torch::Tensor output =
      torch::as_strided(input, /*size=*/size, /*stride=*/stride);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output =
        torch::as_strided(xla_input, /*size=*/size, /*stride=*/stride);
    AllClose(output, xla_output);
  });
  for (auto& name : torch_xla::runtime::metrics::GetCounterNames()) {
    std::cout << name << std::endl;
  }
  std::cout << std::endl;
  for (auto& name : torch::lazy::GetCounterNames()) {
    std::cout << name << std::endl;
  }
  ExpectCounterNotChanged("aten::*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::take", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::as_strided_copy", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAsStridedMultipleDimMismatch) {
  torch::lazy::MetricsArena::Get()->ResetMetrics();
  runtime::metrics::ClearMetrics();
  torch::Tensor input =
      torch::rand({6, 4, 2, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> size = {6, 4, 1, 2};
  std::vector<int64_t> stride = {32, 8, 8, 2};
  torch::Tensor output =
      torch::as_strided(input, /*size=*/size, /*stride=*/stride);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output =
        torch::as_strided(xla_input, /*size=*/size, /*stride=*/stride);
    AllClose(output, xla_output);
  });
  for (auto& name : torch_xla::runtime::metrics::GetCounterNames()) {
    std::cout << name << std::endl;
  }
  std::cout << std::endl;
  for (auto& name : torch::lazy::GetCounterNames()) {
    std::cout << name << std::endl;
  }
  ExpectCounterNotChanged("aten::*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::take", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::as_strided_copy", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAvgPool2DBackward) {
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::avg_pool2d(inputs[0],
                                     /*kernel_size=*/{kernel_size, kernel_size},
                                     /*stride=*/{stride, stride},
                                     /*padding=*/{padding, padding},
                                     /*ceil_mode=*/ceil_mode,
                                     /*count_include_pad=*/count_include_pad);
          };

          ForEachDevice([&](const torch::Device& device) {
            TestBackward(
                {torch::rand(
                    {4, 1, 28, 28},
                    torch::TensorOptions(torch::kFloat).requires_grad(true))},
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
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::avg_pool3d(
                inputs[0],
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
          };

          ForEachDevice([&](const torch::Device& device) {
            TestBackward(
                {torch::rand(
                    {4, 1, 28, 28, 28},
                    torch::TensorOptions(torch::kFloat).requires_grad(true))},
                device, testfn);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAvgPool2DNoBatchBackward) {
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::avg_pool2d(inputs[0],
                                     /*kernel_size=*/{kernel_size, kernel_size},
                                     /*stride=*/{stride, stride},
                                     /*padding=*/{padding, padding},
                                     /*ceil_mode=*/ceil_mode,
                                     /*count_include_pad=*/count_include_pad);
          };

          ForEachDevice([&](const torch::Device& device) {
            TestBackward(
                {torch::rand(
                    {1, 28, 28},
                    torch::TensorOptions(torch::kFloat).requires_grad(true))},
                device, testfn);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAvgPool3DNoBatchBackward) {
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::avg_pool3d(
                inputs[0],
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
          };

          ForEachDevice([&](const torch::Device& device) {
            TestBackward(
                {torch::rand(
                    {1, 28, 28, 28},
                    torch::TensorOptions(torch::kFloat).requires_grad(true))},
                device, testfn);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAdaptiveAvgPool3DNoBatchBackward) {
  for (int64_t output_size : {7, 4}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::adaptive_avg_pool3d(
          inputs[0], {output_size, output_size, output_size});
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {1, 56, 28, 28},
              torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device, testfn);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_adaptive_avg_pool3d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAdaptiveAvgPool3DBackward) {
  for (int64_t output_size : {7, 4}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::adaptive_avg_pool3d(
          inputs[0], {output_size, output_size, output_size});
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {4, 1, 56, 28, 28},
              torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device, testfn);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_adaptive_avg_pool3d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAdaptiveAvgPool2DBackward) {
  for (int64_t output_size : {7, 8}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::adaptive_avg_pool2d(inputs[0], {output_size, output_size});
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {4, 1, 56, 56},
              torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device, testfn);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_adaptive_avg_pool2d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAdaptiveAvgPool2DNoBatchBackward) {
  for (int64_t output_size : {7, 8}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::adaptive_avg_pool2d(inputs[0], {output_size, output_size});
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward({torch::rand({1, 56, 56}, torch::TensorOptions(torch::kFloat)
                                                 .requires_grad(true))},
                   device, testfn);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_adaptive_avg_pool2d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestConv3DBackward) {
  int in_channels = 4;
  int out_channels = 8;
  int kernel_size = 5;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 1; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        for (int dilation = 1; dilation <= 2; ++dilation) {
          for (int groups :
               {1, 2, 4}) {  // covers normal, grouped, depthwise conv.
            auto testfn =
                [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
              return torch::conv3d(inputs[0], inputs[1], inputs[2],
                                   /*stride=*/{stride, stride, stride},
                                   /*padding=*/{padding, padding, padding},
                                   /*dilation=*/{dilation, dilation, dilation},
                                   groups);
            };

            ForEachDevice([&](const torch::Device& device) {
              torch::Tensor bias =
                  with_bias ? torch::rand({out_channels},
                                          torch::TensorOptions(torch::kDouble))
                            : torch::Tensor();
              TestBackward({torch::rand({4, in_channels, 14, 14, 14},
                                        torch::TensorOptions(torch::kDouble)
                                            .requires_grad(true)),
                            torch::rand({out_channels, in_channels / groups,
                                         kernel_size, kernel_size, kernel_size},
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

}  // namespace cpp_test
}  // namespace torch_xla
