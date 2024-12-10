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

TEST_F(AtenXlaTensorTest, TestNe) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::ne(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::ne(xla_a, xla_b);
    AllEqual(c, xla_c);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::ne", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestNeInplace) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor a_copy = a.clone();
  torch::Tensor b = a.clone();
  b[0] += 1;
  a.ne_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a_copy, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    xla_a.ne_(xla_b);
    AllClose(a, xla_a);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::ne", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEq) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.clone();
  torch::Tensor c = torch::eq(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::eq(xla_a, xla_b);
    AllEqual(c, xla_c);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::eq", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEqInplace) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.clone();
  b[0] += 1;
  torch::Tensor a_copy = a.clone();
  a.eq_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a_copy, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    xla_a.eq_(xla_b);
    AllClose(xla_a, a);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::eq", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestGe) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.clone();
  torch::Tensor c = torch::ge(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::ge(xla_a, xla_b);
    AllEqual(c, xla_c);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::ge", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestGeInplace) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  torch::Tensor a_copy = a.clone();
  a.ge_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a_copy, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    xla_a.ge_(xla_b);
    AllClose(xla_a, a);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::ge", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLe) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.clone();
  torch::Tensor c = torch::le(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::le(xla_a, xla_b);
    AllEqual(c, xla_c);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::le", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLeInplace) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  torch::Tensor a_copy = a.clone();
  a.le_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a_copy, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    xla_a.le_(xla_b);
    AllClose(xla_a, a);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::le", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestGt) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::add(a.clone(), torch::ones_like(a));
  torch::Tensor c = torch::gt(b, a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::gt(xla_b, xla_a);
    AllEqual(c, xla_c);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::gt", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestGtInplace) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  torch::Tensor a_copy = a.clone();
  a.gt_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a_copy, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    xla_a.gt_(xla_b);
    AllClose(xla_a, a);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::gt", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLt) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::add(a.clone(), torch::ones_like(a));
  torch::Tensor c = torch::lt(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::lt(xla_a, xla_b);
    AllEqual(c, xla_c);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::lt", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLtInplace) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  torch::Tensor a_copy = a.clone();
  a.lt_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a_copy, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    xla_a.lt_(xla_b);
    AllClose(xla_a, a);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::lt", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestNeScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(0));
  torch::Tensor result = torch::ne(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::ne(xla_input, other);
    AllEqual(result, xla_result);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::ne", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEqScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(1));
  torch::Tensor result = torch::eq(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::eq(xla_input, other);
    AllEqual(result, xla_result);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::eq", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestGeScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(1));
  torch::Tensor result = torch::ge(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::ge(xla_input, other);
    AllEqual(result, xla_result);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::ge", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestGeScalarInplace) {
  torch::Tensor input =
      torch::arange(-1., 1.5, 0.5, torch::TensorOptions(torch::kFloat));
  torch::Scalar other(float(0));
  torch::Tensor input_copy = input.clone();
  input.ge_(other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input_copy, device);
    xla_input.ge_(other);
    AllClose(xla_input, input);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::ge", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLeScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(1));
  torch::Tensor result = torch::le(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::le(xla_input, other);
    AllEqual(result, xla_result);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::le", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLeScalarInplace) {
  torch::Tensor input =
      torch::arange(-1., 1.5, 0.5, torch::TensorOptions(torch::kFloat));
  torch::Scalar other(float(0));
  torch::Tensor input_copy = input.clone();
  input.le_(other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input_copy, device);
    xla_input.le_(other);
    AllClose(xla_input, input);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::le", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestGtScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(0.5));
  torch::Tensor result = torch::gt(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::gt(xla_input, other);
    AllEqual(result, xla_result);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::gt", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestGtScalarInplace) {
  torch::Tensor input =
      torch::arange(-1., 1.5, 0.5, torch::TensorOptions(torch::kFloat));
  torch::Scalar other(float(0));
  torch::Tensor input_copy = input.clone();
  input.gt_(other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input_copy, device);
    xla_input.gt_(other);
    AllClose(xla_input, input);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::gt", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLtScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(1.5));
  torch::Tensor result = torch::lt(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::lt(xla_input, other);
    AllEqual(result, xla_result);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::lt", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLtScalarInplace) {
  torch::Tensor input =
      torch::arange(-1., 1.5, 0.5, torch::TensorOptions(torch::kFloat));
  torch::Scalar other(float(0));
  torch::Tensor input_copy = input.clone();
  input.lt_(other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input_copy, device);
    xla_input.lt_(other);
    AllClose(xla_input, input);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::lt", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestIntegerAdd) {
  std::vector<torch::ScalarType> types(
      {torch::kByte, torch::kChar, torch::kShort, torch::kInt, torch::kLong});

  ForEachDevice([&](const torch::Device& device) {
    for (auto type : types) {
      torch::Tensor a =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      torch::Tensor b =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      torch::Scalar one =
          isIntegralType(type) ? torch::Scalar(1) : torch::Scalar(1.0);
      torch::Tensor c = torch::add(b, one);

      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = torch::add(xla_b, one);

      AllEqual(c, xla_c);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestSVD) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (auto n : dims) {
      torch::Tensor a =
          torch::rand({m, n}, torch::TensorOptions(torch::kFloat));
      auto b = torch::svd(a, /*some=*/true, /*compute_uv=*/true);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        auto xla_b = torch::svd(xla_a, /*some=*/true, /*compute_uv=*/true);
        // The U and V matrices might have different sign for column vectors, so
        // cannot be compared if not by absolute value.
        AllClose(std::get<0>(b).abs(), std::get<0>(xla_b).abs(), /*rtol=*/1e-3,
                 /*atol=*/1e-4);
        torch::Tensor diag = std::get<1>(b);
        torch::Tensor xla_diag = std::get<1>(xla_b);
        ASSERT_EQ(diag.sizes(), xla_diag.sizes());
        AllClose(diag, xla_diag, /*rtol=*/1e-3,
                 /*atol=*/1e-4);
        AllClose(std::get<2>(b).abs(), std::get<2>(xla_b).abs(), /*rtol=*/1e-3,
                 /*atol=*/1e-4);
      });
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_linalg_svd", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLinalgSVD) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (auto n : dims) {
      torch::Tensor a =
          torch::rand({m, n}, torch::TensorOptions(torch::kFloat));
      auto b =
          torch::_linalg_svd(a, /*full_matrices=*/false, /*compute_uv=*/true);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        auto xla_b = torch::_linalg_svd(xla_a, /*full_matrices=*/false,
                                        /*compute_uv=*/true);
        // The U and V matrices might have different sign for column vectors, so
        // cannot be compared if not by absolute value.
        AllClose(std::get<0>(b).abs(), std::get<0>(xla_b).abs(), /*rtol=*/1e-3,
                 /*atol=*/1e-4);
        torch::Tensor diag = std::get<1>(b);
        torch::Tensor xla_diag = std::get<1>(xla_b);
        ASSERT_EQ(diag.sizes(), xla_diag.sizes());
        AllClose(diag, xla_diag, /*rtol=*/1e-3,
                 /*atol=*/1e-4);
        AllClose(std::get<2>(b).abs(), std::get<2>(xla_b).abs(), /*rtol=*/1e-3,
                 /*atol=*/1e-4);
      });
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_linalg_svd", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLinalgVectorNorm) {
  torch::Tensor a = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  std::vector<float> ords = {0.0, 1.5, std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()};
  for (auto ord : ords) {
    torch::Tensor a_vn = torch::linalg_vector_norm(a, ord);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_a_vn = torch::linalg_vector_norm(xla_a, ord);
      AllClose(a_vn, xla_a_vn, /*rtol=*/1e-3,
               /*atol=*/1e-4);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::linalg_vector_norm",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLinalgVectorNormInDim) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::linalg_vector_norm(a, 2, {dim}, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b =
          torch::linalg_vector_norm(xla_a, 2, {dim}, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::linalg_vector_norm",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLinalgVectorNormInDims) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    torch::Tensor b = torch::linalg_vector_norm(a, 2, dims, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b =
          torch::linalg_vector_norm(xla_a, 2, dims, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::linalg_vector_norm",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLinalgVectorNormInDimsKeep) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    torch::Tensor b = torch::linalg_vector_norm(a, 2, dims, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b =
          torch::linalg_vector_norm(xla_a, 2, dims, /*keepdim=*/true);
      AllClose(b, xla_b);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::linalg_vector_norm",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLinalgVectorNormInDimsKeepDtype) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    torch::Tensor b =
        torch::linalg_vector_norm(a, 2, dims,
                                  /*keepdim=*/true, /*dtype=*/torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b =
          torch::linalg_vector_norm(xla_a, 2, dims,
                                    /*keepdim=*/true, /*dtype=*/torch::kDouble);
      AllClose(b, xla_b);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::linalg_vector_norm",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLinalgEigh) {
  // Hardcode the test input to avoid numerical instability from randomness,
  // which is a problem in eigenvalue decomposition.
  auto complex64 = [](float real, float imag) {
    return c10::complex<float>{real, imag};
  };
  torch::Tensor input = torch::tensor({
      {complex64(1, 0), complex64(2, -7), complex64(4, -8)},
      {complex64(2, 7), complex64(3, 0), complex64(5, -9)},
      {complex64(4, 8), complex64(5, 9), complex64(6, 0)},
  });
  for (std::string_view uplo : {"U", "L"}) {
    auto [eigenvalues, eigenvectors] = torch::linalg_eigh(input, uplo);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      auto [xla_eigenvalues, xla_eigenvectors] = torch::linalg_eigh(xla_input);
      AllClose(eigenvalues, xla_eigenvalues);
      // The eigenvectors of a symmetric matrix are not unique, nor are they
      // continuous with respect to A. Due to this lack of uniqueness, different
      // hardware and software may compute different eigenvectors. Therefore we
      // instead verify that the decomposition follows the mathematical
      // definition.
      torch::Tensor input_reconstructed = torch::mm(
          torch::mm(
              eigenvectors,
              torch::diag(eigenvalues).toType(c10::ScalarType::ComplexFloat)),
          eigenvectors.t().conj());
      auto xla_eigenvalues_cpu = ToCpuTensor(xla_eigenvalues);
      auto xla_eigenvectors_cpu = ToCpuTensor(xla_eigenvectors);
      torch::Tensor xla_input_reconstructed =
          torch::mm(torch::mm(xla_eigenvectors_cpu,
                              torch::diag(xla_eigenvalues_cpu)
                                  .toType(c10::ScalarType::ComplexFloat)),
                    xla_eigenvectors_cpu.t().conj());
      AllClose(input_reconstructed, input);
      AllClose(xla_input_reconstructed, input);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_linalg_eigh", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestQR) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (auto n : dims) {
      torch::Tensor a =
          torch::rand({m, n}, torch::TensorOptions(torch::kFloat));
      auto b = torch::qr(a);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        auto xla_b = torch::qr(xla_a);
        AllClose(std::get<0>(b).abs(), std::get<0>(xla_b).abs(), /*rtol=*/1e-3,
                 /*atol=*/1e-4);
        AllClose(std::get<1>(b).abs(), std::get<1>(xla_b).abs(), /*rtol=*/1e-3,
                 /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestCholesky) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (bool upper : {true, false}) {
      torch::Tensor a =
          torch::rand({3, m, m}, torch::TensorOptions(torch::kFloat));
      torch::Tensor pd_a = torch::matmul(a, torch::transpose(a, 1, 2)) +
                           torch::eye(m, torch::TensorOptions(torch::kFloat));
      auto b = torch::cholesky(pd_a, upper);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(pd_a, device);
        auto xla_b = torch::cholesky(xla_a, upper);
        AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-4);
      });
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::cholesky", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLogDet) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    torch::Tensor a =
        torch::rand({3, m, m}, torch::TensorOptions(torch::kFloat));
    torch::Tensor pd_a = torch::matmul(a, torch::transpose(a, 1, 2)) +
                         torch::eye(m, torch::TensorOptions(torch::kFloat));
    torch::Tensor b = torch::logdet(pd_a);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(pd_a, device);
      torch::Tensor xla_b = torch::logdet(xla_a);
      AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-4);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestSLogDet) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    torch::Tensor a =
        torch::rand({3, m, m}, torch::TensorOptions(torch::kFloat));
    torch::Tensor pd_a = torch::matmul(a, torch::transpose(a, 1, 2)) +
                         torch::eye(m, torch::TensorOptions(torch::kFloat));
    auto b = torch::slogdet(pd_a);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(pd_a, device);
      auto xla_b = torch::slogdet(xla_a);
      AllClose(std::get<0>(b), std::get<0>(xla_b), /*rtol=*/1e-3,
               /*atol=*/1e-4);
      AllClose(std::get<1>(b), std::get<1>(xla_b), /*rtol=*/1e-3,
               /*atol=*/1e-4);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_linalg_slogdet", cpp_test::GetIgnoredCounters());
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
                torch::Tensor a =
                    torch::randn({m, m}, torch::TensorOptions(torch::kFloat));
                torch::Tensor b =
                    torch::randn({m, n}, torch::TensorOptions(torch::kFloat));
                a = batched_a ? a.expand({3, m, m}).clone() : a;
                b = batched_b ? b.expand({3, m, n}).clone() : b;
                auto result = torch::triangular_solve(
                    b, a, /*upper=*/upper, /*transpose=*/transpose,
                    /*unitriangular=*/unitriangular);
                ForEachDevice([&](const torch::Device& device) {
                  torch::Tensor xla_a = CopyToDevice(a, device);
                  torch::Tensor xla_b = CopyToDevice(b, device);
                  auto xla_result = torch::triangular_solve(
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
  torch::Tensor a = torch::rand({4, 5, 3}, torch::TensorOptions(torch::kFloat));
  for (int k = 1; k <= 3; ++k) {
    int rank = a.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      for (bool keepdim : {false, true}) {
        auto b = torch::kthvalue(a, k, dim, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          auto xla_b = torch::kthvalue(xla_a, k, dim, keepdim);
          AllClose(std::get<0>(b), std::get<0>(xla_b));
          AllEqual(std::get<1>(b), std::get<1>(xla_b));
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestTopK) {
  torch::Tensor a = torch::rand({4, 5, 3}, torch::TensorOptions(torch::kFloat));
  for (int k = 1; k <= 3; ++k) {
    int rank = a.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      for (bool largest : {false, true}) {
        auto b = torch::topk(a, k, dim, largest, /*sorted=*/true);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          auto xla_b = torch::topk(xla_a, k, dim, largest, /*sorted=*/true);
          AllClose(std::get<0>(b), std::get<0>(xla_b));
          AllEqual(std::get<1>(b), std::get<1>(xla_b));
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestSort) {
  torch::Tensor a = torch::rand({4, 5, 3}, torch::TensorOptions(torch::kFloat));
  for (int k = 1; k <= 3; ++k) {
    for (int dim = 0; dim < 3; ++dim) {
      for (bool descending : {false, true}) {
        for (bool stable : {false, true}) {
          auto b = torch::sort(a, dim, descending, stable);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_a = CopyToDevice(a, device);
            auto xla_b = torch::sort(xla_a, dim, descending, stable);
            AllClose(std::get<0>(b), std::get<0>(xla_b));
            AllEqual(std::get<1>(b), std::get<1>(xla_b));
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestSortDescWithMinValue) {
  std::vector<int8_t> values{-128, 100};
  torch::Tensor input =
      torch::tensor(values, torch::TensorOptions(torch::kChar));
  auto output = torch::sort(input, /*dim=*/0, /*descending=*/true);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    auto xla_output = torch::sort(xla_input, /*dim=*/0, /*descending=*/true);
    AllEqual(std::get<0>(output), std::get<0>(xla_output));
    AllEqual(std::get<1>(output), std::get<1>(xla_output));
  });
}

TEST_F(AtenXlaTensorTest, TestArgSort) {
  torch::Tensor a = torch::rand({4, 5, 3}, torch::TensorOptions(torch::kFloat));
  for (int k = 1; k <= 3; ++k) {
    for (int dim = 0; dim < 3; ++dim) {
      for (bool descending : {false, true}) {
        torch::Tensor b = torch::argsort(a, dim, descending);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          torch::Tensor xla_b = torch::argsort(xla_a, dim, descending);
          AllEqual(b, xla_b);
        });
      }
    }
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::sort", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMin) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::min(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::min(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMish) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::mish(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::mish(xla_input);
    AllClose(output, xla_output, /*rtol=*/1e-4);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::mish", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMax) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::max(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::max(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestUnaryMin) {
  torch::Tensor input =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::min(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::min(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestUnaryMax) {
  torch::Tensor input =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::max(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::max(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestAll) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor a =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 4}, torch::TensorOptions(scalar_type));
    torch::Tensor b = torch::all(a);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::all(xla_a);
      EqualValues(b, xla_b);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::all", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAllDim) {
  torch::Tensor a =
      torch::randint(0, 5, {2, 3, 4}, torch::TensorOptions(torch::kByte));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::all(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::all(xla_a, dim, /*keepdim=*/false);
      EqualValues(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestAllDimKeep) {
  torch::Tensor a =
      torch::randint(0, 5, {2, 3, 4}, torch::TensorOptions(torch::kByte));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::all(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::all(xla_a, dim, /*keepdim=*/true);
      EqualValues(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestAmax) {
  torch::Tensor input =
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (bool keepdim : {false, true}) {
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor values = torch::amax(input, {dim}, /*keepdim=*/keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_values =
            torch::amax(xla_input, {dim}, /*keepdim=*/keepdim);
        AllClose(values, xla_values);
      });
    }
    for (int dim1 = -rank; dim1 < rank; ++dim1) {
      for (int dim2 = -rank; dim2 < rank; ++dim2) {
        if ((dim1 == dim2) || (dim1 == rank + dim2) || (dim2 == rank + dim1))
          continue;
        torch::Tensor values =
            torch::amax(input, {dim1, dim2}, /*keepdim=*/keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_input = CopyToDevice(input, device);
          torch::Tensor xla_values =
              torch::amax(xla_input, {dim1, dim2}, /*keepdim=*/keepdim);
          AllClose(values, xla_values);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::amax", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAmin) {
  torch::Tensor input =
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (bool keepdim : {false, true}) {
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor values = torch::amin(input, {dim}, /*keepdim=*/keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_values =
            torch::amin(xla_input, {dim}, /*keepdim=*/keepdim);
        AllClose(values, xla_values);
      });
    }
    for (int dim1 = -rank; dim1 < rank; ++dim1) {
      for (int dim2 = -rank; dim2 < rank; ++dim2) {
        if ((dim1 == dim2) || (dim1 == rank + dim2) || (dim2 == rank + dim1))
          continue;
        torch::Tensor values =
            torch::amin(input, {dim1, dim2}, /*keepdim=*/keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_input = CopyToDevice(input, device);
          torch::Tensor xla_values =
              torch::amin(xla_input, {dim1, dim2}, /*keepdim=*/keepdim);
          AllClose(values, xla_values);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::amin", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAny) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor a =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 4}, torch::TensorOptions(scalar_type));
    torch::Tensor b = torch::any(a);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::any(xla_a);
      EqualValues(b, xla_b);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::any", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAnyDim) {
  torch::Tensor a =
      torch::randint(0, 5, {2, 3, 4}, torch::TensorOptions(torch::kByte));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::any(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::any(xla_a, dim, /*keepdim=*/false);
      EqualValues(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestAnyDimKeep) {
  torch::Tensor a =
      torch::randint(0, 5, {2, 3, 4}, torch::TensorOptions(torch::kByte));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::any(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::any(xla_a, dim, /*keepdim=*/true);
      EqualValues(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMean) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::mean(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::mean(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestMeanCast) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::mean(a, torch::kDouble);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::mean(xla_a, torch::kDouble);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestMeanInDim) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::mean(a, {dim});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::mean(xla_a, {dim});
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMeanInDims) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::mean(a, dims);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::mean(xla_a, dims);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMeanInDimsKeepCast) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::mean(a, dims, true, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::mean(xla_a, dims, true, torch::kDouble);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestStd) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto unbiased : {true, false}) {
    torch::Tensor b = torch::std(a, unbiased);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::std(xla_a, unbiased);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestStdInDim) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim();
  for (auto unbiased : {true, false}) {
    for (auto keepdim : {true, false}) {
      for (int dim = -rank; dim < rank; ++dim) {
        torch::Tensor b = torch::std(a, {dim}, unbiased, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          torch::Tensor xla_b = torch::std(xla_a, {dim}, unbiased, keepdim);
          AllClose(b, xla_b);
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestStdWithCorrection) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim();
  std::optional<torch::Scalar> corrections[] = {1, 2, 1.3, std::nullopt};
  for (const auto& correction : corrections) {
    for (auto keepdim : {true, false}) {
      for (const auto& dim :
           std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
        torch::Tensor b = torch::std(a, dim, correction, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          torch::Tensor xla_b = torch::std(xla_a, dim, correction, keepdim);
          AllClose(b, xla_b);
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestStdMeanWithCorrection) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim();
  std::optional<torch::Scalar> corrections[] = {1, 2, 1.3, std::nullopt};
  for (const auto& correction : corrections) {
    for (auto keepdim : {true, false}) {
      for (const auto& dim :
           std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
        auto b = torch::std_mean(a, dim, correction, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          auto xla_b = torch::std_mean(xla_a, dim, correction, keepdim);
          AllClose(std::get<0>(b), std::get<0>(xla_b));
          AllClose(std::get<1>(b), std::get<1>(xla_b));
        });
      }
    }
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::std_mean", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSum) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::sum(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sum(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestSumCast) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::sum(a, torch::kDouble);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sum(xla_a, torch::kDouble);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestSumU8) {
  torch::Tensor a = torch::ones({256}, torch::TensorOptions(torch::kByte));
  torch::Tensor b = torch::sum(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sum(xla_a);
    AllEqual(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestSumInDim) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::sum(a, {dim});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::sum(xla_a, {dim});
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestSumInDims) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::sum(a, dims);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::sum(xla_a, dims);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestSumInDimsKeep) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::sum(a, dims, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::sum(xla_a, dims, /*keepdim=*/true);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestSumInDimsKeepCast) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::sum(a, dims, /*keepdim=*/true, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b =
          torch::sum(xla_a, dims, /*keepdim=*/true, torch::kDouble);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestVar) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (bool unbiased : {true, false}) {
    torch::Tensor b = torch::var(a, unbiased);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::var(xla_a, unbiased);
      AllClose(b, xla_b);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::var", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestVarWithDim) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (bool keepDim : {true, false}) {
      for (bool unbiased : {true, false}) {
        torch::Tensor b = torch::var(a, dims, unbiased, keepDim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          torch::Tensor xla_b = torch::var(xla_a, dims, unbiased, keepDim);
          AllClose(b, xla_b);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::var", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestVarWithCorrection) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  std::optional<torch::Scalar> corrections[] = {1, 2, 1.3, std::nullopt};
  for (const auto& dim : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (bool keepDim : {true, false}) {
      for (const auto& correction : corrections) {
        torch::Tensor b = torch::var(a, dim, correction, keepDim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          torch::Tensor xla_b = torch::var(xla_a, dim, correction, keepDim);
          AllClose(b, xla_b);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::var", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestVarMeanWithCorrection) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  std::optional<torch::Scalar> corrections[] = {1, 2, 1.3, std::nullopt};
  for (const auto& dim : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (const auto& correction : corrections) {
      for (auto keepdim : {true, false}) {
        auto b = torch::var_mean(a, dim, correction, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          auto xla_b = torch::var_mean(xla_a, dim, correction, keepdim);
          AllClose(std::get<0>(b), std::get<0>(xla_b));
          AllClose(std::get<1>(b), std::get<1>(xla_b));
        });
      }
    }
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::var_mean", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMaxInDim) {
  torch::Tensor input =
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    for (bool keepdim : {false, true}) {
      auto values_indices = torch::max(input, dim, /*keepdim=*/keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        auto xla_values_indices =
            torch::max(xla_input, dim, /*keepdim=*/keepdim);
        AllClose(std::get<0>(values_indices), std::get<0>(xla_values_indices));
        AllEqual(std::get<1>(values_indices), std::get<1>(xla_values_indices));
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMinInDim) {
  torch::Tensor input =
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    for (bool keepdim : {false, true}) {
      auto values_indices = torch::min(input, dim, /*keepdim=*/keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        auto xla_values_indices =
            torch::min(xla_input, dim, /*keepdim=*/keepdim);
        AllClose(std::get<0>(values_indices), std::get<0>(xla_values_indices));
        AllEqual(std::get<1>(values_indices), std::get<1>(xla_values_indices));
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestNorm) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::norm(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::norm(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestNormInDim) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::norm(a, 2, {dim}, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::norm(xla_a, 2, {dim}, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestNormInDims) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    torch::Tensor b = torch::norm(a, 2, dims, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::norm(xla_a, 2, dims, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestNormInDimsKeep) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    torch::Tensor b = torch::norm(a, 2, dims, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::norm(xla_a, 2, dims, /*keepdim=*/true);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestNormalTwoTensor) {
  at::Tensor mean = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  at::Tensor std = at::ones({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    at::Tensor xla_mean = bridge::CreateXlaTensor(mean, device);
    at::Tensor xla_std = bridge::CreateXlaTensor(std, device);
    at::Tensor xla_normal = at::normal(xla_mean, xla_std);
    double res_mean = xla_normal.mean().item().toDouble();
    double res_std = xla_normal.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(AtenXlaTensorTest, TestNormalDoubleMean) {
  at::Tensor std = at::ones({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    at::Tensor xla_std = bridge::CreateXlaTensor(std, device);
    at::Tensor xla_normal = at::normal(0, xla_std);
    double res_mean = xla_normal.mean().item().toDouble();
    double res_std = xla_normal.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(AtenXlaTensorTest, TestNormalDoubleStd) {
  at::Tensor mean = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    at::Tensor xla_mean = bridge::CreateXlaTensor(mean, device);
    at::Tensor xla_normal = at::normal(xla_mean, 1);
    double res_mean = xla_normal.mean().item().toDouble();
    double res_std = xla_normal.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(AtenXlaTensorTest, TestNormalInPlace) {
  at::Tensor a = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    xla_a.normal_(/*mean=*/0, /*std=*/1);
    double res_mean = xla_a.mean().item().toDouble();
    double res_std = xla_a.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(AtenXlaTensorTest, TestUniformInPlace) {
  const double eps = 1e-3;
  at::Tensor a = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    xla_a.uniform_(/*from=*/0, /*to=*/1);
    at::Tensor cpu_a = ToCpuTensor(xla_a);
    double res_min = cpu_a.min().item().toDouble();
    double res_max = cpu_a.max().item().toDouble();
    EXPECT_GT(res_min, 0.0 - eps);
    EXPECT_LT(res_max, 1.0 + eps);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::uniform.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestRandomInPlace) {
  for (auto dtype : {torch::kFloat, torch::kDouble, torch::kByte, torch::kChar,
                     torch::kShort, torch::kInt, torch::kLong}) {
    const double eps = 0.15;
    torch::Tensor a = torch::zeros({10, 10, 10}, torch::TensorOptions(dtype));
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      xla_a.random_(/*from=*/0, /*to=*/10);
      double res_mean = xla_a.sum().item().toDouble() / a.numel();
      double res_min = xla_a.min().item().toDouble();
      double res_max = xla_a.max().item().toDouble();
      EXPECT_GT(res_mean, 4.5 - eps);
      EXPECT_LT(res_mean, 4.5 + eps);
      EXPECT_EQ(res_min, 0.0);
      EXPECT_EQ(res_max, 9.0);
    });
  }

  ExpectCounterNotChanged("aten::(?!_local_scalar_dense).*",
                          cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::random_", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestRandomInPlaceDefaultFrom) {
  for (auto dtype : {torch::kFloat, torch::kDouble, torch::kByte, torch::kChar,
                     torch::kShort, torch::kInt, torch::kLong}) {
    const double eps = 0.15;
    torch::Tensor a = torch::zeros({10, 10, 10}, torch::TensorOptions(dtype));
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      xla_a.random_(/*to=*/10);
      double res_mean = xla_a.sum().item().toDouble() / a.numel();
      double res_min = xla_a.min().item().toDouble();
      double res_max = xla_a.max().item().toDouble();
      EXPECT_GT(res_mean, 4.5 - eps);
      EXPECT_LT(res_mean, 4.5 + eps);
      EXPECT_EQ(res_min, 0.0);
      EXPECT_EQ(res_max, 9.0);
    });
  }

  ExpectCounterNotChanged("aten::(?!_local_scalar_dense).*",
                          cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::random_", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestNormGeneral) {
  torch::Tensor a =
      torch::randn({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::norm(a, 3.5);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::norm(xla_a, 3.5);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestNormNuclear) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::norm(a, 1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::norm(xla_a, 1);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestFrobeniusNormInDim) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::frobenius_norm(a, {dim}, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b =
          torch::frobenius_norm(xla_a, {dim}, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestFrobeniusNormInDims) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    torch::Tensor b = torch::frobenius_norm(a, dims, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b =
          torch::frobenius_norm(xla_a, dims, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestGroupNorm) {
  int num_channels = 6;
  torch::Tensor input = torch::rand({20, num_channels, 10, 10},
                                    torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({num_channels}, torch::TensorOptions(torch::kFloat));
  torch::Tensor bias =
      torch::rand({num_channels}, torch::TensorOptions(torch::kFloat));
  double eps = 1e-05;
  for (int num_groups : {3, 6, 1}) {
    torch::Tensor output =
        torch::group_norm(input, num_groups, weight, bias, eps,
                          /*cudnn_enabled=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_weight = CopyToDevice(weight, device);
      torch::Tensor xla_bias = CopyToDevice(bias, device);
      torch::Tensor xla_output =
          torch::group_norm(xla_input, num_groups, xla_weight, xla_bias, eps,
                            /*cudnn_enabled=*/false);
      AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
    });

    ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::native_batch_norm",
                         cpp_test::GetIgnoredCounters());
  }
}

TEST_F(AtenXlaTensorTest, TestGroupNormBackward) {
  int num_channels = 6;
  torch::Tensor input =
      torch::rand({20, num_channels, 10, 10},
                  torch::TensorOptions(torch::kFloat).requires_grad(true));
  torch::Tensor weight = torch::rand(
      {num_channels}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  torch::Tensor bias = torch::rand(
      {num_channels}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  double eps = 1e-05;
  for (bool undef_weight : {true, false}) {
    for (int num_groups : {3, 6, 1}) {
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::group_norm(
            /*input=*/inputs[0], num_groups, inputs[1], inputs[2],
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
      };
      torch::Tensor undef;
      ForEachDevice({XlaDeviceType::CUDA, XlaDeviceType::TPU},
                    [&](const torch::Device& device) {
                      TestBackward({input, undef_weight ? undef : weight,
                                    undef_weight ? undef : bias},
                                   device, testfn,
                                   /*rtol=*/1e-3, /*atol=*/1e-3,
                                   /*derivative_level=*/2);
                      ExpectCounterNotChanged("aten::.*",
                                              cpp_test::GetIgnoredCounters());
                      ExpectCounterChanged("xla::native_batch_norm",
                                           cpp_test::GetIgnoredCounters());
                      ExpectCounterChanged("xla::native_batch_norm_backward",
                                           cpp_test::GetIgnoredCounters());
                    });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestInstanceNorm) {
  int batch = 5;
  int num_channels = 20;
  torch::Tensor input = torch::rand({batch, num_channels, 10, 10},
                                    torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({num_channels}, torch::TensorOptions(torch::kFloat));
  torch::Tensor bias =
      torch::rand({num_channels}, torch::TensorOptions(torch::kFloat));
  torch::Tensor running_mean =
      torch::zeros({num_channels}, torch::TensorOptions(torch::kFloat));
  torch::Tensor running_var =
      torch::ones({num_channels}, torch::TensorOptions(torch::kFloat));
  double momentum = 0.1;
  double eps = 1e-05;
  torch::Tensor output = torch::instance_norm(
      input, weight, bias, running_mean, running_var,
      /*use_input_stats=*/true, momentum, eps, /*cudnn_enabled=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_weight = CopyToDevice(weight, device);
    torch::Tensor xla_bias = CopyToDevice(bias, device);
    torch::Tensor xla_running_mean = CopyToDevice(running_mean, device);
    torch::Tensor xla_running_var = CopyToDevice(running_var, device);
    torch::Tensor xla_output = torch::instance_norm(
        xla_input, xla_weight, xla_bias, xla_running_mean, xla_running_var,
        /*use_input_stats=*/true, momentum, eps, /*cudnn_enabled=*/false);
    AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::native_batch_norm",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLayerNorm) {
  torch::Tensor input =
      torch::rand({20, 10, 10, 10}, torch::TensorOptions(torch::kFloat));
  double eps = 1e-05;
  torch::Tensor undef;
  for (bool undef_weight : {true, false}) {
    for (int64_t normalized_size : {2, 3}) {
      std::vector<int64_t> normalized_shape(normalized_size, 10);
      torch::Tensor weight =
          torch::rand(normalized_shape, torch::TensorOptions(torch::kFloat));
      torch::Tensor bias =
          torch::rand(normalized_shape, torch::TensorOptions(torch::kFloat));
      torch::Tensor output = torch::layer_norm(input, normalized_shape,
                                               undef_weight ? undef : weight,
                                               undef_weight ? undef : bias, eps,
                                               /*cudnn_enabled=*/false);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_weight =
            undef_weight ? undef : CopyToDevice(weight, device);
        torch::Tensor xla_bias =
            undef_weight ? undef : CopyToDevice(bias, device);
        torch::Tensor xla_output = torch::layer_norm(
            xla_input, normalized_shape, xla_weight, xla_bias, eps,
            /*cudnn_enabled=*/false);
        AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::native_batch_norm",
                           cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestLayerNormBackward) {
  torch::Tensor input = torch::rand(
      {2, 3, 3, 3}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  double eps = 1e-05;
  for (bool undef_weight : {true, false}) {
    for (int64_t normalized_size : {2, 3}) {
      std::vector<int64_t> normalized_shape(normalized_size, 3);
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::layer_norm(
            /*input=*/inputs[0], normalized_shape, inputs[1], inputs[2],
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
      };
      torch::Tensor weight =
          torch::rand(normalized_shape,
                      torch::TensorOptions(torch::kFloat).requires_grad(true));
      torch::Tensor bias =
          torch::rand(normalized_shape,
                      torch::TensorOptions(torch::kFloat).requires_grad(true));
      torch::Tensor undef;
      ForEachDevice([&](const torch::Device& device) {
        TestBackward(
            {input, undef_weight ? undef : weight, undef_weight ? undef : bias},
            device, testfn,
            /*rtol=*/1e-3, /*atol=*/1e-4, /*derivative_level=*/2);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::native_batch_norm",
                           cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::native_batch_norm_backward",
                           cpp_test::GetIgnoredCounters());
    }
  }
}

// TEST_F(AtenXlaTensorTest, TestNuclearNorm) {
//   torch::Tensor a = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
//   torch::Tensor b = torch::nuclear_norm(a);
//   ForEachDevice([&](const torch::Device& device) {
//     torch::Tensor xla_a = CopyToDevice(a, device);
//     torch::Tensor xla_b = torch::nuclear_norm(xla_a);
//     AllClose(b, xla_b);
//   });

//   ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
//   ExpectCounterChanged("xla::svd", cpp_test::GetIgnoredCounters());
// }

TEST_F(AtenXlaTensorTest, TestPairwiseDistance) {
  torch::Tensor x1 = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor x2 = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  double eps = 1e-6;
  for (bool keepdim : {false, true}) {
    for (double p : {1, 2, 3, 4}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output =
            torch::pairwise_distance(x1, x2, p, eps, keepdim);
        torch::Tensor xla_x1 = CopyToDevice(x1, device);
        torch::Tensor xla_x2 = CopyToDevice(x2, device);
        torch::Tensor xla_output =
            torch::pairwise_distance(xla_x1, xla_x2, p, eps, keepdim);
        AllClose(output, xla_output, /*rtol=*/1e-5, /*atol=*/1e-5);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::norm", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestCosineSimilarity) {
  torch::Tensor x1 = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor x2 = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  double eps = 1e-8;
  int rank = x1.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor output = torch::cosine_similarity(x1, x2, dim, eps);
      torch::Tensor xla_x1 = CopyToDevice(x1, device);
      torch::Tensor xla_x2 = CopyToDevice(x2, device);
      torch::Tensor xla_output =
          torch::cosine_similarity(xla_x1, xla_x2, dim, eps);
      AllClose(output, xla_output);
    });

    ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::sum", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::clamp_min", cpp_test::GetIgnoredCounters());
  }
}

TEST_F(AtenXlaTensorTest, TestCosineEmbeddingLoss) {
  torch::Tensor input1 =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor input2 =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output = torch::cosine_embedding_loss(
            input1, input2, target, margin, reduction);
        torch::Tensor xla_input1 = CopyToDevice(input1, device);
        torch::Tensor xla_input2 = CopyToDevice(input2, device);
        torch::Tensor xla_target = CopyToDevice(target, device);
        torch::Tensor xla_output = torch::cosine_embedding_loss(
            xla_input1, xla_input2, xla_target, margin, reduction);
        AllClose(output, xla_output);
      });
      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::clamp_min", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestHingeEmbeddingLoss) {
  torch::Tensor input =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output =
            torch::hinge_embedding_loss(input, target, margin, reduction);
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_target = CopyToDevice(target, device);
        torch::Tensor xla_output = torch::hinge_embedding_loss(
            xla_input, xla_target, margin, reduction);
        AllClose(output, xla_output);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::clamp_min", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestTripletMarginLoss) {
  torch::Tensor anchor =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor positive =
      torch::abs(torch::rand({4, 3}, torch::TensorOptions(torch::kFloat)));
  torch::Tensor negative = torch::neg(
      torch::abs(torch::rand({4, 3}, torch::TensorOptions(torch::kFloat))));
  double eps = 1e-6;
  for (double margin : {0., 0.2}) {
    for (double p : {1, 2, 3, 4}) {
      for (bool swap : {false, true}) {
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean, torch::Reduction::Sum}) {
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor output = torch::triplet_margin_loss(
                anchor, positive, negative, margin, p, eps, swap, reduction);
            torch::Tensor xla_anchor = CopyToDevice(anchor, device);
            torch::Tensor xla_positive = CopyToDevice(positive, device);
            torch::Tensor xla_negative = CopyToDevice(negative, device);
            torch::Tensor xla_output = torch::triplet_margin_loss(
                xla_anchor, xla_positive, xla_negative, margin, p, eps, swap,
                reduction);
            AllClose(output, xla_output);
          });

          ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
          ExpectCounterChanged("xla::clamp_min",
                               cpp_test::GetIgnoredCounters());
          ExpectCounterChanged("xla::norm", cpp_test::GetIgnoredCounters());
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestBinaryCrossEntropy) {
  int batch = 10;
  int classes = 5;
  torch::Tensor input =
      torch::rand({batch, classes}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target =
      torch::rand({batch, classes}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({batch, classes}, torch::TensorOptions(torch::kFloat));
  torch::Tensor undef;
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum,
        torch::Reduction::None}) {
    for (bool undef_weight : {false, true}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output = torch::binary_cross_entropy(
            input, target, undef_weight ? undef : weight, reduction);
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_target = CopyToDevice(target, device);
        torch::Tensor xla_weight =
            undef_weight ? undef : CopyToDevice(weight, device);
        torch::Tensor xla_output = torch::binary_cross_entropy(
            xla_input, xla_target, xla_weight, reduction);
        AllClose(output, xla_output, /*rtol=*/1e-4, /*atol=*/1e-5);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::binary_cross_entropy",
                           cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMarginRankingLoss) {
  torch::Tensor input1 =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor input2 =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output = torch::margin_ranking_loss(
            input1, input2, target, margin, reduction);
        torch::Tensor xla_input1 = CopyToDevice(input1, device);
        torch::Tensor xla_input2 = CopyToDevice(input2, device);
        torch::Tensor xla_target = CopyToDevice(target, device);
        torch::Tensor xla_output = torch::margin_ranking_loss(
            xla_input1, xla_input2, xla_target, margin, reduction);
        AllClose(output, xla_output);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::clamp_min", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestBCEWithLogits) {
  int batch = 10;
  int classes = 5;
  torch::Tensor input =
      torch::rand({batch, classes}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target =
      torch::rand({batch, classes}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({classes}, torch::TensorOptions(torch::kFloat));
  torch::Tensor pos_weight =
      torch::rand({classes}, torch::TensorOptions(torch::kFloat));
  torch::Tensor undef;
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    for (bool undef_weight : {false, true}) {
      for (bool undef_pos_weight : {false, true}) {
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor output = torch::binary_cross_entropy_with_logits(
              input, target, undef_weight ? undef : weight,
              undef_pos_weight ? undef : pos_weight, reduction);
          torch::Tensor xla_input = CopyToDevice(input, device);
          torch::Tensor xla_target = CopyToDevice(target, device);
          torch::Tensor xla_weight =
              undef_weight ? undef : CopyToDevice(weight, device);
          torch::Tensor xla_pos_weight =
              undef_pos_weight ? undef : CopyToDevice(pos_weight, device);
          torch::Tensor xla_output = torch::binary_cross_entropy_with_logits(
              xla_input, xla_target, xla_weight, xla_pos_weight, reduction);
        });

        ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
        // binary_cross_entropy_with_logits is composed of
        // sub/mul_/add_/exp_/add_/log_/... ops in upstream pytorch.
        ExpectCounterChanged("xla::mul", cpp_test::GetIgnoredCounters());
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestKlDiv) {
  torch::Tensor input =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  for (bool log_target : {true, false}) {
    for (torch::Reduction::Reduction reduction :
         {torch::Reduction::Mean, torch::Reduction::Sum}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output =
            torch::kl_div(input, target, reduction, log_target);
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_target = CopyToDevice(target, device);
        torch::Tensor xla_output =
            torch::kl_div(xla_input, xla_target, reduction, log_target);
        AllClose(output, xla_output);
      });
      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::mul", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::sub", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestProd) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::prod(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::prod(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestProdCast) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::prod(a, torch::kDouble);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::prod(xla_a, torch::kDouble);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestProdInDim) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::prod(a, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::prod(xla_a, dim);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestProdInDimKeepCast) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::prod(a, dim, /*keepdim=*/true, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b =
          torch::prod(xla_a, dim, /*keepdim=*/true, torch::kDouble);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestProdInDimKeep) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::prod(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::prod(xla_a, dim, /*keepdim=*/true);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCumSum) {
  torch::Tensor input =
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::cumsum(xla_input, dim);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCumSumCast) {
  torch::Tensor input =
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::cumsum(xla_input, dim, torch::kDouble);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCumSumLong) {
  torch::Tensor input =
      torch::randint(1000, {4, 3, 4}, torch::TensorOptions(torch::kLong));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::cumsum(xla_input, dim);
      AllEqual(result, xla_result);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::cumsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestCumSumCastLong) {
  torch::Tensor input =
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim, torch::kLong);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::cumsum(xla_input, dim, torch::kLong);
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCumProd) {
  torch::Tensor input =
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumprod(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::cumprod(xla_input, dim);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCumProdCast) {
  torch::Tensor input = torch::mul(
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat)), 10);
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumprod(input, dim, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::cumprod(xla_input, dim, torch::kDouble);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCumProdLong) {
  torch::Tensor input =
      torch::randint(7, {2, 3}, torch::TensorOptions(torch::kLong));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::cumsum(xla_input, dim);
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCumProdCastLong) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat)) * 7;
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim, torch::kLong);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::cumsum(xla_input, dim, torch::kLong);
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestArgMin) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::argmin(a, std::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::argmin(xla_a, std::nullopt, /*keepdim=*/false);
    AllEqual(b, xla_b);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::argmin", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestArgMinDim) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmin(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::argmin(xla_a, dim, /*keepdim=*/false);
      AllEqual(b, xla_b);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::argmin", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestArgMinDimKeep) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmin(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::argmin(xla_a, dim, /*keepdim=*/true);
      AllEqual(b, xla_b);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::argmin", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestArgMinDimKeepNoDim) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::argmin(a, std::nullopt, /*keepdim=*/true);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::argmin(xla_a, std::nullopt, /*keepdim=*/true);
    AllEqual(b, xla_b);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::argmin", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestArgMinSameValue) {
  torch::Tensor a = torch::ones({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::argmin(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::argmin(xla_a);
    AllEqual(b, xla_b);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::argmin", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestArgMinWrapper) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmin(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::argmin(xla_a, dim, /*keepdim=*/false);
      AllEqual(b, xla_b);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::argmin", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestArgMax) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::argmax(a, std::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::argmax(xla_a, std::nullopt, /*keepdim=*/false);
    AllEqual(b, xla_b);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::argmax", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestArgMaxDim) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmax(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::argmax(xla_a, dim, /*keepdim=*/false);
      AllEqual(b, xla_b);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::argmax", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestArgMaxDimKeep) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmax(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::argmax(xla_a, dim, /*keepdim=*/true);
      AllEqual(b, xla_b);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::argmax", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestArgMaxDimKeepNoDim) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::argmax(a, std::nullopt, /*keepdim=*/true);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::argmax(xla_a, std::nullopt, /*keepdim=*/true);
    AllEqual(b, xla_b);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::argmax", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestArgMaxSameValue) {
  torch::Tensor a = torch::ones({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::argmax(a, std::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::argmax(xla_a, std::nullopt, /*keepdim=*/false);
    AllEqual(b, xla_b);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::argmax", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestArgMaxWrapper) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmax(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::argmax(xla_a, dim, /*keepdim=*/false);
      AllEqual(b, xla_b);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::argmax", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAsin) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::asin(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::asin(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestAsinhWithInt) {
  torch::Tensor a = torch::rand({2, 2});
  torch::Tensor b = torch::asinh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::asinh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::asinh", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAsinh) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::asinh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::asinh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::asinh", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAsinhInPlace) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = torch::asinh_(a);
    torch::Tensor xla_b = torch::asinh_(xla_a);
    AllClose(a, xla_a, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::asinh", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSin) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::sin(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sin(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestSinh) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::sinh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sinh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestAcos) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::acos(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::acos(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

// In torch, acos works with integer inputs. The same should be true for
// torch_xla
TEST_F(AtenXlaTensorTest, TestAcosWithInt) {
  torch::Tensor a = torch::rand({2, 2});
  torch::Tensor b = torch::acos(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::acos(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::acos", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAcosh) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100;
  torch::Tensor b = torch::acosh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::acosh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::acosh", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAcoshInPlace) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = torch::acosh_(a);
    torch::Tensor xla_b = torch::acosh_(xla_a);
    AllClose(a, xla_a, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::acosh", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestCos) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::cos(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::cos(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestCosh) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::cosh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::cosh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestAtan) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::atan(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::atan(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestAtanh) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::atanh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::atanh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::atanh", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAtanhInPlace) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = torch::atanh_(a);
    torch::Tensor xla_b = torch::atanh_(xla_a);
    AllClose(a, xla_a, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::atanh", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAtan2) {
  torch::Tensor a = torch::randn({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::randn({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::atan2(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::atan2(xla_a, xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestTan) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::tan(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::tan(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::tan", cpp_test::GetIgnoredCounters());
}

// In torch, tan works with integer inputs. The same should be true for
// torch_xla
TEST_F(AtenXlaTensorTest, TestTanWithInt) {
  torch::Tensor a = torch::rand({2, 2});
  torch::Tensor b = torch::tan(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::tan(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::tan", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestTanh) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::tanh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::tanh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

// In torch, tanh works with integer inputs. The same should be true for
// torch_xla
TEST_F(AtenXlaTensorTest, TestTanhWithInt) {
  torch::Tensor a = torch::rand({2, 2});
  torch::Tensor b = torch::tanh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::tanh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::tanh", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestClampMinMax) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar min_val(0.311);
  torch::Scalar max_val(0.409);
  torch::Tensor b = torch::clamp(a, min_val, max_val);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::clamp(xla_a, min_val, max_val);
    AllClose(b, xla_b);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::clamp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestClampMinMaxTensor) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor min_tensor =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor max_tensor = min_tensor + 0.1;
  torch::Tensor b = torch::clamp(a, min_tensor, max_tensor);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_min_tensor = CopyToDevice(min_tensor, device);
    torch::Tensor xla_max_tensor = CopyToDevice(max_tensor, device);
    torch::Tensor xla_b = torch::clamp(xla_a, xla_min_tensor, xla_max_tensor);
    AllClose(b, xla_b);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::clamp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestClampMin) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar min_val(0.311);
  torch::Tensor b = torch::clamp(a, min_val, std::nullopt);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::clamp(xla_a, min_val, std::nullopt);
    AllClose(b, xla_b);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::clamp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestClampMinTensor) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor min_tensor =
      torch::rand({1, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::clamp(a, min_tensor, std::nullopt);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_min_tensor = CopyToDevice(min_tensor, device);
    torch::Tensor xla_b = torch::clamp(xla_a, xla_min_tensor, std::nullopt);
    AllClose(b, xla_b);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::clamp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestClampMax) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar max_val(0.409);
  torch::Tensor b = torch::clamp(a, std::nullopt, max_val);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::clamp(xla_a, std::nullopt, max_val);
    AllClose(b, xla_b);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::clamp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestClampMaxTensor) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor max_tensor =
      torch::rand({2, 1}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::clamp(a, std::nullopt, max_tensor);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_max_tensor = CopyToDevice(max_tensor, device);
    torch::Tensor xla_b = torch::clamp(xla_a, std::nullopt, xla_max_tensor);
    AllClose(b, xla_b);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::clamp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestClampMinExplicit) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar min_val(0.311);
  torch::Tensor b = torch::clamp_min(a, min_val);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::clamp_min(xla_a, min_val);
    AllClose(b, xla_b);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::clamp_min", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestClampMinTensorExplicit) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor min_tensor =
      torch::rand({1, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::clamp_min(a, min_tensor);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_min_tensor = CopyToDevice(min_tensor, device);
    torch::Tensor xla_b = torch::clamp_min(xla_a, xla_min_tensor);
    AllClose(b, xla_b);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::clamp_min", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestClampMaxExplicit) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar max_val(0.409);
  torch::Tensor b = torch::clamp_max(a, max_val);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::clamp_max(xla_a, max_val);
    AllClose(b, xla_b);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::clamp_max", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestClampMaxTensorExplicit) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor max_tensor =
      torch::rand({1, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::clamp_max(a, max_tensor);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_max_tensor = CopyToDevice(max_tensor, device);
    torch::Tensor xla_b = torch::clamp_max(xla_a, xla_max_tensor);
    AllClose(b, xla_b);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::clamp_max", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestClampMinExplicitInPlace) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar min_val(0.311);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = torch::clamp_min_(a, min_val);
    torch::Tensor xla_b = torch::clamp_min_(xla_a, min_val);
    AllClose(a, xla_a);
    AllClose(b, xla_b);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::clamp_min", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestClampMaxExplicitInPlace) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar max_val(0.409);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = torch::clamp_max_(a, max_val);
    torch::Tensor xla_b = torch::clamp_max_(xla_a, max_val);
    AllClose(a, xla_a);
    AllClose(b, xla_b);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::clamp_max", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestCeil) {
  torch::Tensor a =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::ceil(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::ceil(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestFloor) {
  torch::Tensor a =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::floor(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::floor(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestFloorDivide) {
  for (torch::ScalarType scalar_type1 : {torch::kFloat, torch::kInt}) {
    torch::Tensor a =
        isFloatingType(scalar_type1)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1)) - 0.5f
            : torch::randint(0, 100, {3, 4},
                             torch::TensorOptions(scalar_type1));
    for (torch::ScalarType scalar_type2 : {torch::kFloat, torch::kInt}) {
      torch::Tensor b =
          isFloatingType(scalar_type2)
              ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2)) + 0.5f
              : torch::randint(1, 100, {3, 4},
                               torch::TensorOptions(scalar_type2));
      torch::Tensor c = torch::floor_divide(a, b);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::Tensor xla_b = CopyToDevice(b, device);
        torch::Tensor xla_c = torch::floor_divide(xla_a, xla_b);
        AllClose(c, xla_c);
      });
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::div", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestRound) {
  torch::Tensor a = torch::cat(
      {torch::randn({8}, torch::TensorOptions(torch::kFloat)) * 100.0,
       // Special case: 0.5, -0.5. xla::Round impl rounds to -1/1 whereas
       // xla::RoundToEven properly implements bankers rounding.
       torch::tensor({-0.5, 0.5}, torch::TensorOptions(torch::kFloat))},
      0);
  torch::Tensor b = torch::round(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::round(xla_a);
    AllClose(b, xla_b);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::round", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestTrunc) {
  torch::Tensor a =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::trunc(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::trunc(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestFrac) {
  torch::Tensor a =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::frac(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::frac(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestNeg) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::neg(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::neg(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseNot) {
  if (UsingTpu()) {
    GTEST_SKIP();
  }

  std::vector<torch::ScalarType> types(
      {torch::kByte, torch::kChar, torch::kShort, torch::kInt, torch::kLong});

  ForEachDevice([&](const torch::Device& device) {
    for (auto type : types) {
      torch::Tensor a =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      torch::Tensor b = torch::bitwise_not(a);
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::bitwise_not(xla_a);
      AllEqual(b, xla_b);
    }
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBitwiseNotInPlace) {
  std::vector<torch::ScalarType> types(
      {torch::kByte, torch::kChar, torch::kShort, torch::kInt, torch::kLong});

  ForEachDevice([&](const torch::Device& device) {
    for (auto type : types) {
      torch::Tensor a =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      torch::Tensor xla_a = CopyToDevice(a, device);
      a.bitwise_not_();
      xla_a.bitwise_not_();
      AllEqual(a, xla_a);
    }
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSgn) {
  torch::Tensor a =
      torch::randn({2, 2}, torch::TensorOptions(torch::kComplexFloat)) * 100.0;
  torch::Tensor b = torch::sgn(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sgn(xla_a);
    AllClose(b, xla_b);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::sgn", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSign) {
  torch::Tensor a = torch::randn({2, 2}, torch::TensorOptions(torch::kFloat))
                        .mul_(100.0)
                        .set_requires_grad(true);
  torch::Tensor b = torch::sign(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device, /*requires_grad=*/true);
    torch::Tensor xla_b = torch::sign(xla_a);
    AllClose(b, xla_b);
    AssertBackward(xla_b, {xla_a}, b, {a});
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSignByte) {
  torch::Tensor a =
      torch::randint(256, {2, 2}, torch::TensorOptions(torch::kByte));
  torch::Tensor b = torch::sign(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sign(xla_a);
    AllEqual(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestAbs) {
  torch::Tensor a = torch::randn(
      {2, 2}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  torch::Tensor b = torch::abs(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device, /*requires_grad=*/true);
    torch::Tensor xla_b = torch::abs(xla_a);
    AllClose(b, xla_b);
    AssertBackward(xla_b, {xla_a}, b, {a});
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAbsByte) {
  torch::Tensor a =
      torch::randint(256, {2, 2}, torch::TensorOptions(torch::kByte));
  torch::Tensor b = torch::abs(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::abs(xla_a);
    AllEqual(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestEmptyLike) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::empty_like(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::empty_like(xla_a);
    EXPECT_EQ(b.sizes(), xla_b.sizes());
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::empty", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEmptyLikeOptions) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::empty_like(a, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b =
        torch::empty_like(xla_a, torch::TensorOptions(torch::kFloat));
    EXPECT_EQ(b.sizes(), xla_b.sizes());
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::empty", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEmpty) {
  torch::Tensor a = torch::zeros({2, 2}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = torch::empty(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(device));
    EXPECT_EQ(a.sizes(), xla_a.sizes());
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::empty", cpp_test::GetIgnoredCounters());
}

}  // namespace cpp_test
}  // namespace torch_xla
