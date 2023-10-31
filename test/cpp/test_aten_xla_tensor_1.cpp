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

TEST_F(AtenXlaTensorTest, TestStorage) {
  torch::Tensor a = torch::tensor({0.0});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    XLATensorPtr xla_tensor_a = bridge::GetXlaTensor(xla_a);
    EXPECT_EQ(xla_a.device(), xla_tensor_a->Storage().device());
    AllClose(a, xla_a);
  });
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

TEST_F(AtenXlaTensorTest, TestZerosLike) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::zeros_like(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::zeros_like(xla_a);
    AllClose(a, xla_a);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::empty", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::zero_", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestZerosLikeOptions) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::zeros_like(a, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b =
        torch::zeros_like(xla_a, torch::TensorOptions(torch::kFloat));
    AllClose(a, xla_a);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::empty", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_copy_from", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestZeros) {
  torch::Tensor a = torch::zeros({2, 2}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = torch::zeros(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(a, xla_a);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::empty", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::zero_", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestOnes) {
  torch::Tensor a = torch::ones({2, 2}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a =
        torch::ones({2, 2}, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(a, xla_a);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::empty", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::fill_", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestOnesLike) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::ones_like(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::ones_like(xla_a);
    AllClose(a, xla_a);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::empty", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::fill_", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestOnesLikeOptions) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::ones_like(a, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b =
        torch::ones_like(xla_a, torch::TensorOptions(torch::kFloat));
    AllClose(a, xla_a);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::empty", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_copy_from", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestFull) {
  torch::Tensor a =
      torch::full({2, 2}, 3.1165, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = torch::full(
        {2, 2}, 3.1165, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(a, xla_a);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::empty", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::fill_", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestFullLike) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::full_like(a, 3.1165);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::full_like(xla_a, 3.1165);
    AllClose(a, xla_a);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::empty", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::fill_", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestFullLikeOptions) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b =
      torch::full_like(a, 3.1165, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b =
        torch::full_like(xla_a, 3.1165, torch::TensorOptions(torch::kFloat));
    AllClose(a, xla_a);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::empty", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_copy_from", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestARange) {
  for (auto& ranges : std::vector<std::vector<float>>{{0.0, 100.0, 0.5},
                                                      {0.0, -100.0, -0.5}}) {
    torch::Tensor a = torch::arange(ranges[0], ranges[1], ranges[2],
                                    torch::TensorOptions(torch::kFloat));
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a =
          torch::arange(ranges[0], ranges[1], ranges[2],
                        torch::TensorOptions(torch::kFloat).device(device));
      AllClose(a, xla_a);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::arange_out", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestARangeOut) {
  torch::Tensor a = torch::randn({4}, torch::TensorOptions(torch::kFloat));
  for (auto& ranges : std::vector<std::vector<float>>{{0.0, 100.0, 0.5},
                                                      {0.0, -100.0, -0.5}}) {
    torch::Tensor b = torch::arange_out(a, ranges[0], ranges[1], ranges[2]);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b =
          torch::arange_out(xla_a, ranges[0], ranges[1], ranges[2]);
      AllClose(b, xla_b);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::arange_out", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestDimARange) {
  torch::Tensor like = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor a = torch::_dim_arange(like, 1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_like = CopyToDevice(like, device);
    torch::Tensor xla_a = torch::_dim_arange(xla_like, 1);
    AllClose(a, xla_a);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::arange_out", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBartlettWindow) {
  int window_length = 10;
  for (bool periodic : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor output = torch::bartlett_window(
          window_length, periodic, torch::TensorOptions(torch::kFloat));

      torch::Tensor xla_output = torch::bartlett_window(
          window_length, periodic,
          torch::TensorOptions(torch::kFloat).device(device));
      AllClose(output, xla_output, /*rtol=*/1e-5, /*atol=*/1e-7);
    });

    ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::arange_out", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::slice_copy", cpp_test::GetIgnoredCounters());
  }
}

TEST_F(AtenXlaTensorTest, TestBlackmanWindow) {
  int window_length = 10;
  for (bool periodic : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor output = torch::blackman_window(
          window_length, periodic, torch::TensorOptions(torch::kFloat));
      torch::Tensor xla_output = torch::blackman_window(
          window_length, periodic,
          torch::TensorOptions(torch::kFloat).device(device));
      AllClose(output, xla_output, /*rtol=*/1e-5, /*atol=*/1e-7);
    });

    ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::arange_out", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::cos", cpp_test::GetIgnoredCounters());
  }
}

TEST_F(AtenXlaTensorTest, TestHammingWindow) {
  double alpha = 0.54;
  double beta = 0.46;
  int window_length = 10;
  for (bool periodic : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor output =
          torch::hamming_window(window_length, periodic, alpha, beta,
                                torch::TensorOptions(torch::kFloat));
      torch::Tensor xla_output = torch::hamming_window(
          window_length, periodic, alpha, beta,
          torch::TensorOptions(torch::kFloat).device(device));
      AllClose(output, xla_output);
    });

    ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::arange_out", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::cos", cpp_test::GetIgnoredCounters());
  }
}

TEST_F(AtenXlaTensorTest, TestHannWindow) {
  int window_length = 10;
  for (bool periodic : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor output = torch::hann_window(
          window_length, periodic, torch::TensorOptions(torch::kFloat));
      torch::Tensor xla_output = torch::hann_window(
          window_length, periodic,
          torch::TensorOptions(torch::kFloat).device(device));
      AllClose(output, xla_output);
    });

    ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::arange_out", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::cos", cpp_test::GetIgnoredCounters());
  }
}

TEST_F(AtenXlaTensorTest, TestLogSigmoid) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::log_sigmoid(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::log_sigmoid(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::log_sigmoid_forward",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLogsumexp) {
  torch::Tensor a = torch::rand({3, 4, 3}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (bool keepdim : {false, true}) {
      torch::Tensor b = torch::logsumexp(a, dims, keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::Tensor xla_b = torch::logsumexp(xla_a, dims, keepdim);
        AllClose(b, xla_b);
      });
      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::logsumexp", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestXLogY) {
  torch::Tensor a = torch::rand({5, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({5, 5}, torch::TensorOptions(torch::kFloat));
  a[0][0] = 0.0;
  b[0][2] = std::nan("1");
  b[0][0] = std::nan("1");
  torch::Tensor c = torch::xlogy(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::xlogy(xla_a, xla_b);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::xlogy", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSiLU) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::silu(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::silu(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::silu", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSiLUBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::silu(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 2},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::sigmoid", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSigmoid) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::sigmoid(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sigmoid(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestMatmul_1x1) {
  torch::Tensor a = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::matmul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::matmul(xla_a, xla_b);
    AllClose(c, xla_c);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::dot", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMatmul_2x1) {
  torch::Tensor a = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::matmul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::matmul(xla_a, xla_b);
    AllClose(c, xla_c);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::mv", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMatmul_1x2) {
  torch::Tensor a = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::matmul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::matmul(xla_a, xla_b);
    AllClose(c, xla_c);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::mm", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMatmul_2x2) {
  torch::Tensor a = torch::rand({2, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::matmul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::matmul(xla_a, xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-3, /*atol=*/1e-4);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::mm", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMatmulBcast) {
  torch::Tensor a =
      torch::rand({4, 2, 3, 2, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b =
      torch::rand({2, 1, 4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::matmul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::matmul(xla_a, xla_b);
    AllClose(c, xla_c);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestDot) {
  torch::Tensor a = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::dot(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::dot(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestTensorDot) {
  torch::Tensor a = torch::rand({6, 4, 8}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({4, 7, 8}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> dims_a = {1, 2};
  std::vector<int64_t> dims_b = {0, 2};
  torch::Tensor c = torch::tensordot(a, b, dims_a, dims_b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::tensordot(xla_a, xla_b, dims_a, dims_b);
    AllClose(c, xla_c);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::mm", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestGer) {
  torch::Tensor a = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::ger(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::ger(xla_a, xla_b);
    AllClose(c, xla_c);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMv) {
  torch::Tensor a = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::mv(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::mv(xla_a, xla_b);
    AllClose(c, xla_c);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::mv", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMvOut) {
  torch::Tensor a = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({4}, torch::TensorOptions(torch::kFloat));
  torch::mv_out(c, a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::empty({4}, xla_b.options());
    torch::mv_out(xla_c, xla_a, xla_b);
    AllClose(c, xla_c);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::mv", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBatchAddBatchMatMul) {
  torch::Tensor a = torch::rand({3, 6, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 6, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::rand({3, 4, 5}, torch::TensorOptions(torch::kFloat));
  torch::Scalar alpha = 0.5;
  torch::Scalar beta = 1.5;
  torch::Tensor d = torch::baddbmm(a, b, c, beta, alpha);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::baddbmm(xla_a, xla_b, xla_c, beta, alpha);
    AllClose(d, xla_d, /*rtol=*/1e-3, /*atol=*/1e-4);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::baddbmm", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBatchAddBatchMatMulInPlace) {
  torch::Tensor a = torch::rand({3, 6, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 6, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::rand({3, 4, 5}, torch::TensorOptions(torch::kFloat));
  torch::Scalar alpha = 0.5;
  torch::Scalar beta = 1.5;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor d = a.baddbmm_(b, c, beta, alpha);
    torch::Tensor xla_d = xla_a.baddbmm_(xla_b, xla_c, beta, alpha);
    AllClose(d, xla_d, /*rtol=*/1e-3, /*atol=*/1e-4);
    AllClose(a, xla_a, /*rtol=*/1e-3, /*atol=*/1e-4);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::baddbmm", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBatchMatMul) {
  torch::Tensor a = torch::rand({3, 6, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 4, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::bmm(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::bmm(xla_a, xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-3, /*atol=*/1e-4);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bmm", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLinear) {
  torch::Tensor input =
      torch::rand({2, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor bias = torch::rand({3});
  torch::Tensor result = torch::linear(input, weight);
  torch::Tensor result_with_bias = torch::linear(input, weight, bias);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_weight = CopyToDevice(weight, device);
    torch::Tensor xla_bias = CopyToDevice(bias, device);
    torch::Tensor xla_result = torch::linear(xla_input, xla_weight);
    torch::Tensor xla_result_with_bias =
        torch::linear(xla_input, xla_weight, xla_bias);
    AllClose(result, xla_result, /*rtol=*/1e-2, /*atol=*/1e-4);
    AllClose(result_with_bias, xla_result_with_bias, /*rtol=*/1e-2,
             /*atol=*/1e-4);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestPinverse) {
  torch::Tensor input =
      torch::rand({4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Tensor result = torch::pinverse(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::pinverse(xla_input);
    AllClose(result, xla_result, /*rtol=*/1e-4);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_linalg_svd", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumOuter) {
  torch::Tensor a = torch::rand({5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({5}, torch::TensorOptions(torch::kFloat));
  std::string equation = "i,j->ij";
  torch::Tensor c = torch::einsum(equation, {a, b});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::einsum(equation, {xla_a, xla_b});
    AllClose(c, xla_c);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumOuterBackward) {
  torch::Tensor a =
      torch::rand({5}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  torch::Tensor b =
      torch::rand({5}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  std::string equation = "i,j->ij";
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::einsum(equation, inputs);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward({a, b}, device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumBatchMatMul) {
  if (UsingTpu()) {
    GTEST_SKIP();
  }
  torch::Tensor a = torch::rand({3, 2, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 5, 4}, torch::TensorOptions(torch::kFloat));
  std::string equation = "bij,bjk->bik";
  torch::Tensor c = torch::einsum(equation, {a, b});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::einsum(equation, {xla_a, xla_b});
    AllClose(c, xla_c);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumBatchMatMulBackward) {
  if (UsingTpu()) {
    GTEST_SKIP();
  }
  torch::Tensor a = torch::rand(
      {3, 2, 5}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  torch::Tensor b = torch::rand(
      {3, 5, 4}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  std::string equation = "bij,bjk->bik";
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::einsum(equation, inputs);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward({a, b}, device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerBilinear) {
  torch::Tensor a = torch::rand({3, 5, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor l = torch::rand({2, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor r = torch::rand({2, 4}, torch::TensorOptions(torch::kFloat));
  std::string equation = "bn,anm,bm->ba";
  torch::Tensor c = torch::einsum(equation, {l, a, r});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_l = CopyToDevice(l, device);
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_r = CopyToDevice(r, device);
    torch::Tensor xla_c = torch::einsum(equation, {xla_l, xla_a, xla_r});
    AllClose(c, xla_c);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerBilinearBackward) {
  torch::Tensor a = torch::rand(
      {3, 5, 4}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  torch::Tensor l = torch::rand(
      {2, 5}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  torch::Tensor r = torch::rand(
      {2, 4}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  std::string equation = "bn,anm,bm->ba";
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::einsum(equation, inputs);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward({l, a, r}, device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerDiagonal) {
  torch::Tensor input =
      torch::rand({3, 3}, torch::TensorOptions(torch::kFloat));
  std::string equation = "ii->i";
  torch::Tensor result = torch::einsum(equation, {input});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::einsum(equation, {xla_input});
    AllClose(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerDiagonalBackward) {
  torch::Tensor input = torch::rand(
      {3, 3}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  std::string equation = "ii->i";
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::einsum(equation, inputs);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward({input}, device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerBatchDiagonal) {
  torch::Tensor input =
      torch::rand({4, 3, 3}, torch::TensorOptions(torch::kFloat));
  std::string equation = "...ii->...i";
  torch::Tensor result = torch::einsum(equation, {input});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::einsum(equation, {xla_input});
    AllClose(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerBatchDiagonalBackward) {
  torch::Tensor input = torch::rand(
      {4, 3, 3}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  std::string equation = "...ii->...i";
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::einsum(equation, inputs);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward({input}, device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerBatchPermute) {
  torch::Tensor input =
      torch::rand({2, 3, 4, 5}, torch::TensorOptions(torch::kFloat));
  std::string equation = "...ij->...ji";
  torch::Tensor result = torch::einsum(equation, {input});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::einsum(equation, {xla_input});
    AllClose(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerBatchPermuteBackward) {
  torch::Tensor input = torch::rand(
      {2, 3, 4, 5}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  std::string equation = "...ij->...ji";
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::einsum(equation, inputs);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward({input}, device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerRepeatedAxis) {
  torch::Tensor x = torch::rand({2, 3, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor y = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  std::string equation = "ijj,k->ik";
  torch::Tensor result = torch::einsum(equation, {x, y});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_x = CopyToDevice(x, device);
    torch::Tensor xla_y = CopyToDevice(y, device);
    torch::Tensor xla_result = torch::einsum(equation, {xla_x, xla_y});
    AllClose(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerRepeatedAxisBackward) {
  torch::Tensor x = torch::rand(
      {2, 3, 3}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  torch::Tensor y =
      torch::rand({4}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  std::string equation = "ijj,k->ik";
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::einsum(equation, inputs);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward({x, y}, device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumThreeInputs) {
  torch::Tensor x = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor y = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor z = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  std::string equation = "i,j,k->ijk";

  torch::Tensor result = torch::einsum(equation, {x, y, z});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_x = CopyToDevice(x, device);
    torch::Tensor xla_y = CopyToDevice(y, device);
    torch::Tensor xla_z = CopyToDevice(z, device);
    torch::Tensor xla_result = torch::einsum(equation, {xla_x, xla_y, xla_z});
    AllClose(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumExtraSpaces) {
  torch::Tensor a = torch::rand({5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({5}, torch::TensorOptions(torch::kFloat));
  std::string equation = "i, j->ij";
  torch::Tensor result = torch::einsum(equation, {a, b});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_result = torch::einsum(equation, {xla_a, xla_b});
    AllClose(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEinsumLarge4D) {
  if (UsingTpu()) {
    GTEST_SKIP();
  }
  torch::Tensor a =
      torch::rand({8, 16, 1024, 128}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b =
      torch::rand({8, 16, 1024, 128}, torch::TensorOptions(torch::kFloat));

  std::string equation = "ijkl,ijml->ijkm";
  torch::Tensor result = torch::einsum(equation, {a, b});

  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_result = torch::einsum(equation, {xla_a, xla_b});
    AllClose(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("EinsumFallback", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::einsum", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBilinear) {
  int batch_size = 16;
  int in1_features = 4;
  int in2_features = 6;
  int out_features = 8;
  torch::Tensor input1 = torch::rand({batch_size, in1_features},
                                     torch::TensorOptions(torch::kFloat));
  torch::Tensor input2 = torch::rand({batch_size, in2_features},
                                     torch::TensorOptions(torch::kFloat));
  torch::Tensor weight = torch::rand({out_features, in1_features, in2_features},
                                     torch::TensorOptions(torch::kFloat));
  torch::Tensor bias =
      torch::rand({out_features}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input1 = CopyToDevice(input1, device);
    torch::Tensor xla_input2 = CopyToDevice(input2, device);
    torch::Tensor xla_weight = CopyToDevice(weight, device);
    torch::Tensor xla_bias = CopyToDevice(bias, device);
    torch::Tensor result = torch::bilinear(input1, input2, weight, bias);
    torch::Tensor xla_result =
        torch::bilinear(xla_input1, xla_input2, xla_weight, xla_bias);
    AllClose(result, xla_result);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestUpsampleNearest2D) {
  struct ImageInfo {
    int batch_size;
    int h;
    int w;
    int uh;
    int uw;
    int chans;
  };

  /* clang-format off */
  std::vector<ImageInfo> inputs = {
    {/*batch_size=*/2, /*h=*/5, /*w=*/5, /*uh=*/8, /*uw=*/8, /*chans=*/2},
    {/*batch_size=*/2, /*h=*/1335, /*w=*/1335, /*uh=*/255, /*uw=*/255, /*chans=*/3},
    {/*batch_size=*/2, /*h=*/255, /*w=*/255, /*uh=*/1335, /*uw=*/1335, /*chans=*/3},
    {/*batch_size=*/2, /*h=*/254, /*w=*/243, /*uh=*/784, /*uw=*/214, /*chans=*/3}
  };
  /* clang-format on */

  for (const auto& img_info : inputs) {
    torch::Tensor input = torch::rand(
        {img_info.batch_size, img_info.chans, img_info.h, img_info.w},
        torch::TensorOptions(torch::kFloat));
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor result =
          torch::upsample_nearest2d(input, {img_info.uh, img_info.uw});
      torch::Tensor xla_result =
          torch::upsample_nearest2d(xla_input, {img_info.uh, img_info.uw});
      AllClose(result, xla_result);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::upsample_nearest2d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestUpsampleNearest2DBackward) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int uh = 8;
  int uw = 8;
  int chans = 2;
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::upsample_nearest2d(inputs[0], {uh, uw});
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({batch_size, chans, h, w},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
  ExpectCounterChanged("xla::upsample_nearest2d_backward",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestUpsampleNearest2DWithScale) {
  struct ImageInfo {
    int batch_size;
    int h;
    int w;
    int chans;
    double scale_h;
    double scale_w;
  };

  /* clang-format off */
  std::vector<ImageInfo> inputs = {
    {/*batch_size=*/2, /*h=*/5, /*w=*/5, /*chans=*/2, /*scale_h*/2.5, /*scale_w*/3.4},
    {/*batch_size=*/2, /*h=*/1335, /*w=*/1335, /*chans=*/3, /*scale_h*/2.5, /*scale_w*/3.4},
    {/*batch_size=*/2, /*h=*/1335, /*w=*/1335, /*chans=*/3, /*scale_h*/0.5, /*scale_w*/0.5},
  };
  /* clang-format on */

  for (const auto& img_info : inputs) {
    torch::Tensor input = torch::rand(
        {img_info.batch_size, img_info.chans, img_info.h, img_info.w},
        torch::TensorOptions(torch::kFloat));
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor result = torch::upsample_nearest2d(
          input, c10::nullopt,
          at::ArrayRef<double>{img_info.scale_h, img_info.scale_w});
      torch::Tensor xla_result = torch::upsample_nearest2d(
          xla_input, c10::nullopt,
          at::ArrayRef<double>{img_info.scale_h, img_info.scale_w});
      AllClose(result, xla_result);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::upsample_nearest2d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestUpsampleNearest2DBackwardWithScale) {
  struct ImageInfo {
    int batch_size;
    int h;
    int w;
    int chans;
    double scale_h;
    double scale_w;
  };

  /* clang-format off */
  std::vector<ImageInfo> inputs = {
    {/*batch_size=*/2, /*h=*/5, /*w=*/5, /*chans=*/2, /*scale_h*/2.5, /*scale_w*/3.4},
    {/*batch_size=*/2, /*h=*/1335, /*w=*/1335, /*chans=*/3, /*scale_h*/2.5, /*scale_w*/3.4},
    {/*batch_size=*/2, /*h=*/1335, /*w=*/1335, /*chans=*/3, /*scale_h*/0.5, /*scale_w*/0.5},
  };
  /* clang-format on */

  for (const auto& img_info : inputs) {
    for (bool align_corners : {true, false}) {
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::upsample_nearest2d(
            inputs[0], c10::nullopt,
            at::ArrayRef<double>{img_info.scale_h, img_info.scale_w});
      };
      ForEachDevice([&](const torch::Device& device) {
        TestBackward(
            {torch::rand(
                {img_info.batch_size, img_info.chans, img_info.h, img_info.w},
                torch::TensorOptions(torch::kFloat).requires_grad(true))},
            device, testfn);
        XlaDeviceType device_type = static_cast<XlaDeviceType>(
            bridge::AtenDeviceToXlaDevice(device).type());
        if (device_type == XlaDeviceType::TPU) {
          // Only lowered for TPU, fallback for CPU.
          ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
          ExpectCounterChanged("xla::upsample_nearest2d_backward",
                               cpp_test::GetIgnoredCounters());
          ResetCounters();
        } else {
          ExpectCounterChanged("aten::.*", cpp_test::GetIgnoredCounters());
          ResetCounters();
        }
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestUpsampleBilinear2D) {
  struct ImageInfo {
    int batch_size;
    int h;
    int w;
    int uh;
    int uw;
    int chans;
  };

  /* clang-format off */
  std::vector<ImageInfo> inputs = {
    {/*batch_size=*/2, /*h=*/5, /*w=*/5, /*uh=*/8, /*uw=*/8, /*chans=*/2},
    {/*batch_size=*/2, /*h=*/1335, /*w=*/1335, /*uh=*/255, /*uw=*/255, /*chans=*/3},
    {/*batch_size=*/2, /*h=*/255, /*w=*/255, /*uh=*/1335, /*uw=*/1335, /*chans=*/3},
    {/*batch_size=*/2, /*h=*/254, /*w=*/243, /*uh=*/784, /*uw=*/214, /*chans=*/3}
  };
  /* clang-format on */

  for (const auto& img_info : inputs) {
    for (bool align_corners : {true, false}) {
      torch::Tensor input = torch::rand(
          {img_info.batch_size, img_info.chans, img_info.h, img_info.w},
          torch::TensorOptions(torch::kFloat));
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor result = torch::upsample_bilinear2d(
            input, {img_info.uh, img_info.uw}, align_corners);
        torch::Tensor xla_result = torch::upsample_bilinear2d(
            xla_input, {img_info.uh, img_info.uw}, align_corners);
        AllClose(result, xla_result, /*rtol=*/1e-4, /*atol=*/1e-4);
      });
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::upsample_bilinear2d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestUpsampleBilinear2DWithScale) {
  struct ImageInfo {
    int batch_size;
    int h;
    int w;
    int chans;
    double scale_h;
    double scale_w;
  };

  /* clang-format off */
  std::vector<ImageInfo> inputs = {
    {/*batch_size=*/2, /*h=*/5, /*w=*/5, /*chans=*/2, /*scale_h*/8.0/5, /*scale_w*/8.0/5},
    {/*batch_size=*/2, /*h=*/1335, /*w=*/1335, /*chans=*/3, /*scale_h*/255.0/1335, /*scale_w*/255.0/1335},
    {/*batch_size=*/2, /*h=*/255, /*w=*/255, /*chans=*/3, /*scale_h*/1335.0/255, /*scale_w*/1335.0/255},
    {/*batch_size=*/2, /*h=*/254, /*w=*/243, /*chans=*/3, /*scale_h*/784.0/254, /*scale_w*/214.0/243}
  };
  /* clang-format on */

  for (const auto& img_info : inputs) {
    for (bool align_corners : {true, false}) {
      torch::Tensor input = torch::rand(
          {img_info.batch_size, img_info.chans, img_info.h, img_info.w},
          torch::TensorOptions(torch::kFloat));
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor result = torch::upsample_bilinear2d(
            input, c10::nullopt, align_corners,
            at::ArrayRef<double>{img_info.scale_h, img_info.scale_w});
        torch::Tensor xla_result = torch::upsample_bilinear2d(
            xla_input, c10::nullopt, align_corners,
            at::ArrayRef<double>{img_info.scale_h, img_info.scale_w});
        AllClose(result, xla_result, /*rtol=*/1e-4, /*atol=*/1e-4);
      });
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::upsample_bilinear2d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestUpsampleBilinear2DBackward) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int uh = 8;
  int uw = 8;
  int chans = 2;
  for (bool align_corners : {true, false}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::upsample_bilinear2d(inputs[0], {uh, uw}, align_corners);
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {batch_size, chans, h, w},
              torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device, testfn);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestUpsampleBilinear2DBackwardWithScale) {
  struct ImageInfo {
    int batch_size;
    int h;
    int w;
    int chans;
    double scale_h;
    double scale_w;
  };

  /* clang-format off */
  std::vector<ImageInfo> inputs = {
    {/*batch_size=*/2, /*h=*/5, /*w=*/5, /*chans=*/2, /*scale_h*/8.0/5, /*scale_w*/8.0/5},
    {/*batch_size=*/2, /*h=*/1335, /*w=*/1335, /*chans=*/3, /*scale_h*/255.0/1335, /*scale_w*/255.0/1335},
  };
  /* clang-format on */

  for (const auto& img_info : inputs) {
    for (bool align_corners : {true, false}) {
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::upsample_bilinear2d(
            inputs[0], c10::nullopt, align_corners,
            at::ArrayRef<double>{img_info.scale_h, img_info.scale_w});
      };
      ForEachDevice([&](const torch::Device& device) {
        TestBackward(
            {torch::rand(
                {img_info.batch_size, img_info.chans, img_info.h, img_info.w},
                torch::TensorOptions(torch::kFloat).requires_grad(true))},
            device, testfn);
        XlaDeviceType device_type = static_cast<XlaDeviceType>(
            bridge::AtenDeviceToXlaDevice(device).type());
        if (device_type == XlaDeviceType::TPU) {
          // Only lowered for TPU, fallback for CPU.
          ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
          ExpectCounterChanged("xla::upsample_bilinear2d_backward",
                               cpp_test::GetIgnoredCounters());
          ResetCounters();
        } else {
          ExpectCounterChanged("aten::.*", cpp_test::GetIgnoredCounters());
          ResetCounters();
        }
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAddCMul) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor d = torch::addcmul(a, b, c, 3.1165);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::addcmul(xla_a, xla_b, xla_c, 3.1165);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenXlaTensorTest, TestAddCDiv) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c =
      torch::abs(torch::rand({2, 2}, torch::TensorOptions(torch::kFloat))) +
      1.0;
  torch::Tensor d = torch::addcdiv(a, b, c, 3.1165);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::addcdiv(xla_a, xla_b, xla_c, 3.1165);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenXlaTensorTest, TestSize) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    for (int dim = -rank; dim < rank; ++dim) {
      EXPECT_EQ(torch::size(input, dim), torch::size(xla_input, dim));
    }
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  // tensor.size(dim) is tensor property that PyTorch's implementation
  // is common to all devices. So we don't assert postive checks here.
}

TEST_F(AtenXlaTensorTest, TestSelect) {
  torch::Tensor input =
      torch::rand({14, 24, 8}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output = torch::select(input, dim, 4);
      torch::Tensor xla_output = torch::select(xla_input, dim, 4);
      AllClose(output, xla_output);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestBernoulliScalarProb) {
  torch::Tensor input = torch::zeros(1000, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::bernoulli(xla_input, 0.1);
    double frac = xla_output.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });

  ExpectCounterNotChanged("aten::(?!_local_scalar_dense).*",
                          cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bernoulli", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBernoulliTensorProb) {
  std::vector<float> prob_values(1000, 0.1);
  torch::Tensor input =
      torch::tensor(prob_values, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::bernoulli(xla_input);
    double frac = xla_output.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });

  ExpectCounterNotChanged("aten::(?!_local_scalar_dense).*",
                          cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bernoulli", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBernoulliScalarProbInPlace) {
  torch::Tensor input = torch::zeros(1000, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    xla_input.bernoulli_(0.1);
    double frac = xla_input.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
  ExpectCounterNotChanged("aten::(?!_local_scalar_dense).*",
                          cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bernoulli", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBernoulliTensorProbInPlace) {
  torch::Tensor input = torch::zeros(1000, torch::TensorOptions(torch::kFloat));
  torch::Tensor prob =
      torch::scalar_tensor(0.1, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_prob = CopyToDevice(prob, device);
    xla_input.bernoulli_(xla_prob);
    double frac = xla_input.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
  ExpectCounterNotChanged("aten::(?!_local_scalar_dense).*",
                          cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::bernoulli_", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestDropout) {
  torch::Tensor a = torch::rand({17, 21}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::dropout(xla_a, 0.1, /*train=*/true);
    double prob =
        static_cast<double>(xla_b.cpu().ne(0.0f).sum().item().toDouble()) /
        a.numel();
    EXPECT_GT(prob, 0.86);
    EXPECT_LT(prob, 0.94);
  });

  ExpectCounterNotChanged("aten::(?!_local_scalar_dense).*",
                          cpp_test::GetIgnoredCounters());
  // dropout is composed of many arithmetic ops.
  ExpectCounterChanged("xla::bernoulli", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestDropoutInPlace) {
  torch::Tensor a = torch::rand({17, 21}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::dropout_(xla_a, 0.1, /*train=*/true);
    double prob =
        static_cast<double>(xla_a.cpu().ne(0.0f).sum().item().toDouble()) /
        a.numel();
    EXPECT_GT(prob, 0.85);
    EXPECT_LT(prob, 0.94);
  });

  ExpectCounterNotChanged("aten::(?!_local_scalar_dense).*",
                          cpp_test::GetIgnoredCounters());
  // dropout is composed of many arithmetic ops.
  ExpectCounterChanged("xla::bernoulli", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestNativeDropout) {
  torch::Tensor a = torch::rand({17, 21}, torch::TensorOptions(torch::kFloat));
  float allowance = 0.04;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    for (float probability : {0.1, 0.5}) {
      auto [xla_b_val, xla_b_mask] =
          torch::native_dropout(xla_a, probability, /*train=*/true);
      double prob = static_cast<double>(
                        xla_b_val.cpu().eq(0.0f).sum().item().toDouble()) /
                    a.numel();
      EXPECT_GT(prob, probability - allowance);
      EXPECT_LT(prob, probability + allowance);
      EXPECT_EQ(xla_b_val.scalar_type(), torch::kFloat);
      EXPECT_EQ(xla_b_mask.scalar_type(), torch::kBool);
    }
  });

  ExpectCounterNotChanged("aten::(?!_local_scalar_dense).*",
                          cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::native_dropout", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestNativeDropoutNotTrain) {
  torch::Tensor a = torch::rand({17, 21}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    auto [xla_b_val, xla_b_mask] =
        torch::native_dropout(xla_a, 0.5, /*train=*/false);
    AllEqual(xla_b_val, xla_a);
    EXPECT_EQ(xla_b_val.scalar_type(), torch::kFloat);
    EXPECT_EQ(xla_b_mask.scalar_type(), torch::kBool);
  });

  ExpectCounterNotChanged("aten::(?!_local_scalar_dense).*",
                          cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::native_dropout", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestNativeDropoutMask) {
  torch::Tensor a = torch::rand({17, 21}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    auto [xla_b_val, xla_b_mask] =
        torch::native_dropout(xla_a, 0.5, /*train=*/true);
    auto count1 = xla_b_val.cpu().eq(0.0f).sum().item().toInt();
    auto count2 = xla_b_mask.cpu().eq(0.0f).sum().item().toInt();
    EXPECT_EQ(count1, count2);
  });

  ExpectCounterNotChanged("aten::(?!_local_scalar_dense).*",
                          cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::native_dropout", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestNativeDropoutZeroProbability) {
  torch::Tensor a = torch::rand({17, 21}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    auto [xla_b_val, xla_b_mask] =
        torch::native_dropout(xla_a, 0, /*train=*/true);
    auto count1 = xla_b_val.cpu().ne(0.0f).sum().item().toInt();
    auto count2 = xla_b_mask.cpu().ne(0.0f).sum().item().toInt();
    auto count3 = xla_a.cpu().ne(0.0f).sum().item().toInt();
    EXPECT_EQ(count1, count2);
    EXPECT_EQ(count2, count3);
  });

  ExpectCounterNotChanged("aten::(?!_local_scalar_dense).*",
                          cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::native_dropout", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestRandperm) {
  int n = 5;
  torch::Tensor shuffle = torch::randperm(
      n, torch::TensorOptions(torch::kLong).device(torch::kXLA));
  torch::Tensor shuffle_cpu = CopyToDevice(shuffle, torch::kCPU);
  std::vector<int64_t> shuffle_data(shuffle_cpu.data_ptr<int64_t>(),
                                    shuffle_cpu.data_ptr<int64_t>() + n);
  EXPECT_TRUE(shuffle_data.size() == n && xla::IsPermutation(shuffle_data));
  ExpectCounterNotChanged("aten::(?!randperm.generator_out).*",
                          cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSlice) {
  torch::Tensor a =
      torch::rand({32, 24, 16}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::slice(a, 1, 0, 16, 1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::slice(xla_a, 1, 0, 16, 1);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestTake) {
  torch::Tensor a = torch::rand({4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::randint(16, {5}, torch::TensorOptions(torch::kLong));
  torch::Tensor c = torch::take(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::take(xla_a, xla_b);
    AllClose(c, xla_c);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::take", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestTakeBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::take(inputs[0], inputs[1]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({4, 4},
                     torch::TensorOptions(torch::kFloat).requires_grad(true)),
         torch::randint(16, {5}, torch::TensorOptions(torch::kLong))},
        device, testfn);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestStack) {
  torch::Tensor a = torch::rand({2, 4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::rand({2, 4, 3}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim() + 1;
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor d = torch::stack({a, b, c}, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = torch::stack({xla_a, xla_b, xla_c}, dim);
      AllClose(d, xla_d);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCat) {
  torch::Tensor a = torch::rand({2, 1, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::rand({2, 3, 3}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor d = torch::cat({a, b, c}, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = torch::cat({xla_a, xla_b, xla_c}, dim);
      AllClose(d, xla_d);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCatTypePromotion) {
  for (torch::ScalarType scalar_type_1 :
       {torch::kFloat, torch::kDouble, torch::kShort, torch::kInt,
        torch::kLong}) {
    for (torch::ScalarType scalar_type_2 :
         {torch::kFloat, torch::kDouble, torch::kShort, torch::kInt,
          torch::kLong}) {
      torch::Tensor a =
          torch::ones({2, 1, 3}, torch::TensorOptions(scalar_type_1));
      torch::Tensor b =
          torch::ones({2, 2, 3}, torch::TensorOptions(scalar_type_2));
      torch::Tensor c = torch::cat({a, b}, /*dim=*/1);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::Tensor xla_b = CopyToDevice(b, device);
        torch::Tensor xla_c = torch::cat({xla_a, xla_b}, /*dim=*/1);
        EXPECT_EQ(c.scalar_type(), xla_c.scalar_type());
      });
    }
  };
}

TEST_F(AtenXlaTensorTest, TestUnbind) {
  torch::Tensor input =
      torch::rand({4, 3, 7}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    std::vector<torch::Tensor> output = torch::unbind(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      std::vector<torch::Tensor> xla_output = torch::unbind(xla_input, dim);
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
      torch::Tensor input =
          torch::rand(input_size, torch::TensorOptions(torch::kFloat));
      torch::Tensor output = input.repeat(repeats);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_output = xla_input.repeat(repeats);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestGather) {
  torch::Tensor a = torch::rand({3, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::empty({3, 3}, torch::TensorOptions(torch::kLong));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      b[i][j] = (i + j) % 3;
    }
  }
  for (bool sparse_grad : {false, true}) {
    torch::Tensor c = torch::gather(a, 1, b, sparse_grad);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = torch::gather(xla_a, 1, xla_b, sparse_grad);
      AllClose(c, xla_c);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestScatter) {
  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 5}, torch::TensorOptions(torch::kLong));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    torch::Tensor d = torch::scatter(a, dim, c, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = torch::scatter(xla_a, dim, xla_c, xla_b);
      AllClose(d, xla_d);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::scatter", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestScatterR1) {
  torch::Tensor a = torch::rand({5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({2}, torch::TensorOptions(torch::kLong));
  c[0] = 1;
  c[1] = 3;
  torch::Tensor d = torch::scatter(a, 0, c, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::scatter(xla_a, 0, xla_c, xla_b);
    AllClose(d, xla_d);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::scatter", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestScatterR3) {
  torch::Tensor a = torch::rand({3, 5, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 4, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 4, 2}, torch::TensorOptions(torch::kLong));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 2; k++) {
        c[i][j][k] = (i + j + k) % 4;
      }
    }
  }
  torch::Tensor d = torch::scatter(a, 1, c, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::scatter(xla_a, 1, xla_c, xla_b);
    AllClose(d, xla_d);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::scatter", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestScatterBiggerSource) {
  torch::Tensor a = torch::rand({4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({8, 8}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({4, 4}, torch::TensorOptions(torch::kLong));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    torch::Tensor d = torch::scatter(a, dim, c, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = torch::scatter(xla_a, dim, xla_c, xla_b);
      AllClose(d, xla_d);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::scatter", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestScatterScalar) {
  torch::Tensor a = torch::rand({4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Scalar b = 1.0f;
  torch::Tensor c = torch::empty({4, 4}, torch::TensorOptions(torch::kLong));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    torch::Tensor d = torch::scatter(a, dim, c, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = torch::scatter(xla_a, dim, xla_c, b);
      AllClose(d, xla_d);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::scatter", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestScatterReduceAdd) {
  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 5}, torch::TensorOptions(torch::kLong));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    torch::Tensor d = torch::scatter(a, dim, c, b, "add");
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = torch::scatter(xla_a, dim, xla_c, xla_b, "add");
      AllClose(d, xla_d);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::scatter", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestScatterAdd) {
  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 5}, torch::TensorOptions(torch::kLong));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    torch::Tensor d = torch::scatter_add(a, dim, c, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = torch::scatter_add(xla_a, dim, xla_c, xla_b);
      AllClose(d, xla_d);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::scatter_add", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestScatterAddInPlace) {
  torch::Tensor b = torch::rand({4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({4, 4}, torch::TensorOptions(torch::kLong));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor a =
          torch::rand({4, 4}, torch::TensorOptions(torch::kFloat));
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor d = a.scatter_add_(dim, c, b);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = xla_a.scatter_add_(dim, xla_c, xla_b);
      AllClose(d, xla_d);
      AllClose(a, xla_a);
    });

    ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::scatter_add", cpp_test::GetIgnoredCounters());
  }
}

TEST_F(AtenXlaTensorTest, TestScatterReduceSum) {
  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 5}, torch::TensorOptions(torch::kLong));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    torch::Tensor d = torch::scatter_reduce(a, dim, c, b, "sum");
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d =
          torch::scatter_reduce(xla_a, dim, xla_c, xla_b, "sum");
      AllClose(d, xla_d);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::scatter_reduce", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestScatterReduceSumInPlace) {
  torch::Tensor b = torch::rand({4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({4, 4}, torch::TensorOptions(torch::kLong));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor a =
          torch::rand({4, 4}, torch::TensorOptions(torch::kFloat));
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor d = a.scatter_reduce_(dim, c, b, "sum");
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = xla_a.scatter_reduce_(dim, xla_c, xla_b, "sum");
      AllClose(d, xla_d);
      AllClose(a, xla_a);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::scatter_reduce", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestScatterReduceProd) {
  if (UsingTpu()) {
    GTEST_SKIP();
  }

  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 5}, torch::TensorOptions(torch::kLong));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    torch::Tensor d = torch::scatter_reduce(a, dim, c, b, "prod");
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d =
          torch::scatter_reduce(xla_a, dim, xla_c, xla_b, "prod");
      AllClose(d, xla_d);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::scatter_reduce", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestScatterReduceProdInPlace) {
  if (UsingTpu()) {
    GTEST_SKIP();
  }

  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 5}, torch::TensorOptions(torch::kLong));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    torch::Tensor d = torch::scatter_reduce(a, dim, c, b, "prod");
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = xla_a.scatter_reduce(dim, xla_c, xla_b, "prod");
      AllClose(d, xla_d);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::scatter_reduce", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestScatterReduceMin) {
  if (UsingTpu()) {
    GTEST_SKIP();
  }

  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 5}, torch::TensorOptions(torch::kLong));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    torch::Tensor d = torch::scatter_reduce(a, dim, c, b, "amin");
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d =
          torch::scatter_reduce(xla_a, dim, xla_c, xla_b, "amin");
      AllClose(d, xla_d);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::scatter_reduce", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestScatterReduceMinInPlace) {
  if (UsingTpu()) {
    GTEST_SKIP();
  }

  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 5}, torch::TensorOptions(torch::kLong));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    torch::Tensor d = torch::scatter_reduce(a, dim, c, b, "amin");
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = xla_a.scatter_reduce(dim, xla_c, xla_b, "amin");
      AllClose(d, xla_d);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::scatter_reduce", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestScatterReduceMax) {
  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 5}, torch::TensorOptions(torch::kLong));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    torch::Tensor d = torch::scatter_reduce(a, dim, c, b, "amax");
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = scatter_reduce(xla_a, dim, xla_c, xla_b, "amax");
      AllClose(d, xla_d);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::scatter_reduce", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestScatterReduceMaxInPlace) {
  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 5}, torch::TensorOptions(torch::kLong));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    torch::Tensor d = torch::scatter_reduce(a, dim, c, b, "amax");
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = xla_a.scatter_reduce(dim, xla_c, xla_b, "amax");
      AllClose(d, xla_d);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::scatter_reduce", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestUnsafeIndex) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor a =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 4}, torch::TensorOptions(scalar_type));
    for (torch::ScalarType index_scalar_type : {torch::kInt, torch::kLong}) {
      torch::List<torch::optional<torch::Tensor>> indices{
          torch::tensor({0, 1}, torch::TensorOptions(index_scalar_type)),
          torch::tensor({2, 3}, torch::TensorOptions(index_scalar_type))};
      torch::Tensor c0 = torch::_unsafe_index(a, indices);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::List<torch::optional<torch::Tensor>> xla_indices{
            CopyToDevice(*indices.get(0), device),
            CopyToDevice(*indices.get(1), device)};
        torch::Tensor xla_c0 = torch::_unsafe_index(xla_a, xla_indices);
        AllEqual(c0, xla_c0);
      });
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::index", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_unsafe_index", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestIndexSelect) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor a =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 4}, torch::TensorOptions(scalar_type));
    for (torch::ScalarType index_scalar_type : {torch::kInt, torch::kLong}) {
      torch::Tensor b =
          torch::empty({2}, torch::TensorOptions(index_scalar_type));
      b[0] = 0;
      b[1] = 2;
      torch::Tensor c0 = torch::index_select(a, 0, b);
      torch::Tensor c1 = torch::index_select(a, 1, b);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::Tensor xla_b = CopyToDevice(b, device);
        torch::Tensor xla_c0 = torch::index_select(xla_a, 0, xla_b);
        torch::Tensor xla_c1 = torch::index_select(xla_a, 1, xla_b);
        AllEqual(c0, xla_c0);
        AllEqual(c1, xla_c1);
      });
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::index_select", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestIndexSelectRank0) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor a =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 4}, torch::TensorOptions(scalar_type));
    torch::Tensor b =
        torch::scalar_tensor(2, torch::TensorOptions(torch::kLong));
    torch::Tensor c0 = torch::index_select(a, 0, b);
    torch::Tensor c1 = torch::index_select(a, 1, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c0 = torch::index_select(xla_a, 0, xla_b);
      torch::Tensor xla_c1 = torch::index_select(xla_a, 1, xla_b);
      AllEqual(c0, xla_c0);
      AllEqual(c1, xla_c1);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestInverse) {
  torch::Tensor a = torch::randn({5, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::inverse(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::inverse(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
  ExpectCounterNotChanged("aten::(?!_local_scalar_dense).*",
                          cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::linalg_inv_ex", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestIsnan) {
  torch::Tensor a = torch::tensor({1.0, 2.0, std::nan("1"), 4.0},
                                  torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::isnan(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::isnan(xla_a);
    AllEqual(b, xla_b);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::isnan", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestExpand) {
  torch::Tensor a = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.expand({2, 3, 4}, /*implicit=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = xla_a.expand({2, 3, 4}, /*implicit=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestExpandBack) {
  torch::Tensor a = torch::rand({3, 1}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.expand({3, 4}, /*implicit=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = xla_a.expand({3, 4}, /*implicit=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestExpandAs) {
  torch::Tensor a = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::native::expand_as(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::native::expand_as(xla_a, xla_b);
    AllClose(c, xla_c);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::expand_copy", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestExpandSymIntStatic) {
  torch::Tensor a = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.expand({2, 3, 4}, /*implicit=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = xla_a.expand_symint(
        c10::SymIntArrayRef({c10::SymInt(2), c10::SymInt(3), c10::SymInt(4)}),
        /*implicit=*/false);
    AllClose(b, xla_b);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::expand_copy_symint",
                       cpp_test::GetIgnoredCounters());
}

static c10::SymInt make_symint(const torch::lazy::NodePtr& p) {
  return c10::SymInt(static_cast<c10::SymNode>(
      c10::make_intrusive<XLASymNodeImpl>(p, PyType::INT)));
}

TEST_F(AtenXlaTensorTest, TestEye) {
  int n = 5;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor out = torch::eye(n, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_out =
        torch::eye(n, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(out, xla_out);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::eye_out", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEyeWide) {
  int lines = 3;
  int cols = 5;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor out =
        torch::eye(lines, cols, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_out = torch::eye(
        lines, cols, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(out, xla_out);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::eye_out", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEyeNarrow) {
  int lines = 5;
  int cols = 3;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor out =
        torch::eye(lines, cols, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_out = torch::eye(
        lines, cols, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(out, xla_out);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::eye_out", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestBroadcastTensors) {
  torch::Tensor a = torch::rand({2, 1, 1}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 1}, torch::TensorOptions(torch::kFloat));
  std::vector<torch::Tensor> c = torch::broadcast_tensors({a, b});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    std::vector<torch::Tensor> xla_c = torch::broadcast_tensors({xla_a, xla_b});
    ASSERT_EQ(c.size(), xla_c.size());
    for (size_t i = 0; i < c.size(); ++i) {
      AllClose(c[i], xla_c[i]);
    }
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::expand_copy", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestOneIndex) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor indices =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor result = torch::index(params, {indices});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_params = CopyToDevice(params, device);
      torch::Tensor xla_indices = CopyToDevice(indices, device);
      torch::Tensor xla_result = torch::index(xla_params, {xla_indices});
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestOneIndexTransfer) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor indices =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor result = torch::index(params, {indices});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_params = CopyToDevice(params, device);
      torch::Tensor xla_result = torch::index(xla_params, {indices});
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestCount_Nonzero_nodim) {
  torch::Tensor a = torch::zeros({3, 3}, torch::TensorOptions(torch::kFloat));
  a[0][1] = 1.0;
  a[0][2] = 1.0;
  a[2][2] = 1.0;
  torch::Tensor b = torch::count_nonzero(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::count_nonzero(xla_a);
    AllClose(b, torch::_cast_Long(xla_b));
  });
  ExpectCounterChanged("xla::count_nonzero", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestCount_Nonzero_with_single_dim) {
  torch::Tensor a = torch::zeros({3, 3}, torch::TensorOptions(torch::kFloat));
  a[0][1] = 1.0;
  a[0][2] = 1.0;
  a[2][2] = 1.0;
  std::vector<c10::optional<long int>> dims = {0, -1};
  for (int i = 0; i < dims.size(); i++) {
    torch::Tensor b = torch::count_nonzero(a, dims[i]);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::count_nonzero(xla_a, dims[i]);
      AllClose(b, torch::_cast_Long(xla_b));
    });
  }
  ExpectCounterChanged("xla::count_nonzero", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestCount_Nonzero_with_multiple_dims) {
  torch::Tensor a =
      torch::zeros({3, 3, 4}, torch::TensorOptions(torch::kFloat));
  a[0][1][0] = 1.0;
  a[0][2][1] = 1.0;
  a[2][2][2] = 1.0;
  std::vector<long int> dims = {0, 2};
  torch::Tensor b = torch::count_nonzero(a, dims);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::count_nonzero(xla_a, dims);
    AllClose(b, torch::_cast_Long(xla_b));
  });
  ExpectCounterChanged("xla::count_nonzero", cpp_test::GetIgnoredCounters());
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestCount_Nonzero_error_case) {
  torch::Tensor a =
      torch::zeros({3, 3, 4}, torch::TensorOptions(torch::kFloat));
  a[0][1][0] = 1.0;
  a[0][2][1] = 1.0;
  a[2][2][2] = 1.0;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);

    std::vector<long int> dims = {0, 0};
    EXPECT_THROW(torch::count_nonzero(xla_a, dims), std::runtime_error);

    dims = {10};
    EXPECT_THROW(torch::count_nonzero(xla_a, dims), c10::Error);
  });
}

TEST_F(AtenXlaTensorTest, TestNonzero) {
  torch::Tensor a = torch::zeros({4, 2}, torch::TensorOptions(torch::kFloat));
  a[0][1] = 1.0;
  a[1][0] = 2.0;
  a[3][1] = 3.0;
  torch::Tensor b = torch::nonzero(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::nonzero(xla_a);
    AllClose(b, torch::_cast_Long(xla_b));

    if (DebugUtil::ExperimentEnabled("nonzero")) {
      // If the nonzero support is enabled, we must not see any aten:: calls.
      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    }
    ExpectCounterChanged("xla::nonzero", cpp_test::GetIgnoredCounters());
    ResetCounters();
  });
}

TEST_F(AtenXlaTensorTest, TestMaskedSelect) {
  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b =
      torch::randint(0, 2, {5}, torch::TensorOptions(torch::kBool));
  torch::Tensor c = torch::masked_select(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::masked_select(xla_a, xla_b);
    AllClose(c, xla_c);

    if (DebugUtil::ExperimentEnabled("masked_select")) {
      // If the masked_select support is enabled, we must not see any aten::
      // calls.
      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    }
    ExpectCounterChanged("xla::masked_select", cpp_test::GetIgnoredCounters());
    ResetCounters();
  });
}

TEST_F(AtenXlaTensorTest, TestMaskedScatter) {
  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b =
      torch::randint(0, 2, {3, 5}, torch::TensorOptions(torch::kBool));
  torch::Tensor c = torch::rand({15}, torch::TensorOptions(torch::kFloat));
  torch::Tensor d = torch::masked_scatter(a, b, c);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::masked_scatter(xla_a, xla_b, xla_c);
    AllClose(d, xla_d);

    if (DebugUtil::ExperimentEnabled("masked_scatter")) {
      // If the masked_select support is enabled, we must not see any aten::
      // calls.
      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    }
    ExpectCounterChanged("xla::masked_scatter", cpp_test::GetIgnoredCounters());
    ResetCounters();
  });
}

TEST_F(AtenXlaTensorTest, TestMultiIndexHeadNull) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor indices_null;
    torch::Tensor indices_0 =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor indices_1 =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor result =
        torch::index(params, {indices_null, indices_0, indices_1});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_params = CopyToDevice(params, device);
      torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
      torch::Tensor xla_result = torch::index(
          xla_params, {indices_null, xla_indices_0, xla_indices_1});
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexMiddleNull) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor indices_0 =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor indices_null;
    torch::Tensor indices_1 =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor result =
        torch::index(params, {indices_0, indices_null, indices_1});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_params = CopyToDevice(params, device);
      torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
      torch::Tensor xla_result = torch::index(
          xla_params, {xla_indices_0, indices_null, xla_indices_1});
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexTailNull) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor indices_0 =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor indices_null;
    torch::Tensor indices_1 =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor result =
        torch::index(params, {indices_0, indices_1, indices_null});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_params = CopyToDevice(params, device);
      torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
      torch::Tensor xla_result = torch::index(
          xla_params, {xla_indices_0, xla_indices_1, indices_null});
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexMiddleBroadcast) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor indices_0 =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor indices_1 =
        torch::randint(-3, 3, {2, 1, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor result = torch::index(params, {indices_0, indices_1});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_params = CopyToDevice(params, device);
      torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
      torch::Tensor xla_result =
          torch::index(xla_params, {xla_indices_0, xla_indices_1});
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexTailBroadcast) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor indices_0 =
        torch::randint(-3, 3, {2, 1, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor indices_1 =
        torch::randint(-3, 3, {2, 1}, torch::TensorOptions(torch::kLong));
    torch::Tensor result = torch::index(params, {indices_0, indices_1});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_params = CopyToDevice(params, device);
      torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
      torch::Tensor xla_result =
          torch::index(xla_params, {xla_indices_0, xla_indices_1});
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMultinomial) {
  std::vector<int64_t> num_samples = {1, 5};
  std::vector<bool> replacement = {false, true};
  std::vector<std::vector<int64_t>> sizes = {{8}, {6, 4}};
  for (int i = 0; i < num_samples.size(); i++) {
    ForEachDevice([&](const torch::lazy::BackendDevice& device) {
      at::Tensor a = torch::rand(sizes[i], at::dtype(at::kFloat));
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      xla_a.multinomial(num_samples[i], replacement[i]);
      at::Tensor cpu_a = ToCpuTensor(xla_a);
      int64_t res_min = cpu_a.min().item().toLong();
      int64_t res_max = cpu_a.max().item().toLong();
      EXPECT_GE(res_min, 0);
      EXPECT_LT(res_max, sizes[i][0]);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::multinomial.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMaskIndex) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({2, 2}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {2, 2}, torch::TensorOptions(scalar_type));
    torch::Tensor indices =
        torch::randint(0, 2, {2, 2}, torch::TensorOptions(torch::kBool));
    torch::Tensor result = torch::index(params, {indices});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_params = CopyToDevice(params, device);
      torch::Tensor xla_indices = CopyToDevice(indices, device);
      torch::Tensor xla_result = torch::index(xla_params, {xla_indices});
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestOneIndexPut) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor indices =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor values =
        isFloatingType(scalar_type)
            ? torch::rand({3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      torch::Tensor result =
          torch::index_put(params, {indices}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_indices = CopyToDevice(indices, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result =
            torch::index_put(xla_params, {xla_indices}, xla_values, accumulate);
        AllEqual(result, xla_result);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_put_", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestOneIndexPutInPlace) {
  torch::Tensor indices =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor values =
        torch::ones({3, 5, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor params =
            isFloatingType(scalar_type)
                ? torch::rand({4, 3, 5, 6, 7},
                              torch::TensorOptions(scalar_type))
                : torch::randint(100, {4, 3, 5, 6, 7},
                                 torch::TensorOptions(scalar_type));
        torch::Tensor xla_params = CopyToDevice(params.clone(), device);
        torch::Tensor result =
            torch::index_put_(params, {indices}, values, accumulate);
        torch::Tensor xla_indices = CopyToDevice(indices, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result = torch::index_put_(xla_params, {xla_indices},
                                                     xla_values, accumulate);
        AllEqual(result, xla_result);
        AllEqual(params, xla_params);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_put_", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestOneIndexPutTransfer) {
  torch::Tensor indices =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor values =
        torch::ones({3, 5, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      torch::Tensor result =
          torch::index_put(params, {indices}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result =
            torch::index_put(xla_params, {indices}, xla_values, accumulate);
        AllEqual(result, xla_result);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_put_", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexPut) {
  torch::Tensor indices_0 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  torch::Tensor indices_1 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor values =
        torch::ones({5, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      torch::Tensor result =
          torch::index_put(params, {indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result = torch::index_put(
            xla_params, {xla_indices_0, xla_indices_1}, xla_values, accumulate);
        AllEqual(result, xla_result);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_put_", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexPutHeadNull) {
  torch::Tensor indices_0 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  torch::Tensor indices_null;
  torch::Tensor indices_1 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 3, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 3, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor values =
        torch::ones({3, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      torch::Tensor result = torch::index_put(
          params, {indices_null, indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result = torch::index_put(
            xla_params, {indices_null, xla_indices_0, xla_indices_1},
            xla_values, accumulate);
        AllEqual(result, xla_result);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_put_", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexPutMiddleNull) {
  torch::Tensor indices_0 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  torch::Tensor indices_null;
  torch::Tensor indices_1 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 3, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 3, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor values =
        torch::ones({3, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      torch::Tensor result = torch::index_put(
          params, {indices_0, indices_null, indices_1}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result = torch::index_put(
            xla_params, {xla_indices_0, indices_null, xla_indices_1},
            xla_values, accumulate);
        AllEqual(result, xla_result);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_put_", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexPutTailNull) {
  torch::Tensor indices_0 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  torch::Tensor indices_1 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  torch::Tensor indices_null;
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 3, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 3, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor values =
        torch::ones({3, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      torch::Tensor result = torch::index_put(
          params, {indices_0, indices_1, indices_null}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result = torch::index_put(
            xla_params, {xla_indices_0, xla_indices_1, indices_null},
            xla_values, accumulate);
        AllEqual(result, xla_result);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_put_", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexPutMiddleBroadcast) {
  torch::Tensor indices_0 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  torch::Tensor indices_1 =
      torch::randint(-3, 3, {2, 1, 3}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor values =
        torch::ones({5, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      torch::Tensor result =
          torch::index_put(params, {indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result = torch::index_put(
            xla_params, {xla_indices_0, xla_indices_1}, xla_values, accumulate);
        AllEqual(result, xla_result);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_put_", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMultiIndexPutTailBroadcast) {
  torch::Tensor indices_0 =
      torch::randint(-3, 3, {2, 1, 3}, torch::TensorOptions(torch::kLong));
  torch::Tensor indices_1 =
      torch::randint(-3, 3, {2, 1}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor values =
        torch::ones({5, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      torch::Tensor result =
          torch::index_put(params, {indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result = torch::index_put(
            xla_params, {xla_indices_0, xla_indices_1}, xla_values, accumulate);
        AllEqual(result, xla_result);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_put_", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaskIndexPut) {
  torch::Tensor indices =
      torch::tensor({0, 1}, torch::TensorOptions(torch::kByte))
          .to(torch::kBool);
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({2, 2}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {2, 2}, torch::TensorOptions(scalar_type));
    torch::Tensor values = torch::ones({2}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      torch::Tensor result =
          torch::index_put(params, {indices}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_indices = CopyToDevice(indices, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result =
            torch::index_put(xla_params, {xla_indices}, xla_values, accumulate);
        AllEqual(result, xla_result);
      });

      ExpectCounterNotChanged("aten::(?!nonzero).*",
                              cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_put_", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexPutImpl) {
  torch::Tensor indices =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor values =
        torch::ones({3, 5, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor params =
            isFloatingType(scalar_type)
                ? torch::rand({4, 3, 5, 6, 7},
                              torch::TensorOptions(scalar_type))
                : torch::randint(100, {4, 3, 5, 6, 7},
                                 torch::TensorOptions(scalar_type));
        torch::Tensor xla_params = CopyToDevice(params.clone(), device);
        torch::Tensor result = torch::_index_put_impl_(
            params, {indices}, values, accumulate, /*unsafe=*/true);
        torch::Tensor xla_indices = CopyToDevice(indices, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result = torch::_index_put_impl_(
            xla_params, {xla_indices}, xla_values, accumulate, /*unsafe=*/true);
        AllEqual(result, xla_result);
        AllEqual(params, xla_params);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::_index_put_impl_",
                           cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexFillWithScalar) {
  torch::Tensor index =
      torch::tensor({0, 2}, torch::TensorOptions(torch::kLong));
  torch::Scalar value = 42;
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor base =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4, 5}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 4, 5}, torch::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor result = torch::index_fill(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_base = CopyToDevice(base, device);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_result =
            torch::index_fill(xla_base, dim, xla_index, value);
        AllEqual(result, xla_result);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_fill_", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexFillWithScalarInPlace) {
  torch::Tensor index =
      torch::tensor({0, 2}, torch::TensorOptions(torch::kLong));
  torch::Scalar value = 42;
  int rank = 3;
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor base =
            isFloatingType(scalar_type)
                ? torch::rand({3, 4, 5}, torch::TensorOptions(scalar_type))
                : torch::randint(100, {3, 4, 5},
                                 torch::TensorOptions(scalar_type));
        torch::Tensor xla_base = CopyToDevice(base.clone(), device);
        torch::Tensor result = base.index_fill_(dim, index, value);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_result = xla_base.index_fill_(dim, xla_index, value);
        AllEqual(result, xla_result);
        AllEqual(base, xla_base);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_fill_", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexFillWithTensor) {
  torch::Tensor index =
      torch::tensor({0, 2}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor base =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4, 5}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 4, 5}, torch::TensorOptions(scalar_type));
    torch::Tensor value =
        torch::scalar_tensor(42, torch::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor result = torch::index_fill(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_base = CopyToDevice(base, device);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_value = CopyToDevice(value, device);
        torch::Tensor xla_result =
            torch::index_fill(xla_base, dim, xla_index, xla_value);
        AllEqual(result, xla_result);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_fill_", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexFillWithTensorInPlace) {
  torch::Tensor index =
      torch::tensor({0, 2}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor value =
        torch::scalar_tensor(42, torch::TensorOptions(scalar_type));
    int rank = 3;
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor base =
            isFloatingType(scalar_type)
                ? torch::rand({3, 4, 5}, torch::TensorOptions(scalar_type))
                : torch::randint(100, {3, 4, 5},
                                 torch::TensorOptions(scalar_type));
        torch::Tensor xla_base = CopyToDevice(base.clone(), device);
        torch::Tensor result = base.index_fill_(dim, index, value);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_value = CopyToDevice(value, device);
        torch::Tensor xla_result =
            xla_base.index_fill_(dim, xla_index, xla_value);
        AllEqual(result, xla_result);
        AllEqual(base, xla_base);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_fill_", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexFillRank0) {
  torch::Tensor index =
      torch::scalar_tensor(2, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor base =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4, 5}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 4, 5}, torch::TensorOptions(scalar_type));
    torch::Tensor value =
        torch::scalar_tensor(42, torch::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor result = torch::index_fill(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_base = CopyToDevice(base, device);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_value = CopyToDevice(value, device);
        torch::Tensor xla_result =
            torch::index_fill(xla_base, dim, xla_index, xla_value);
        AllEqual(result, xla_result);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_fill_", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexAdd) {
  int index_size = 10;
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor base =
        isFloatingType(scalar_type)
            ? torch::rand({5, 3, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {5, 3, 7}, torch::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      for (torch::ScalarType index_scalar_type : {torch::kInt, torch::kLong}) {
        torch::Tensor index =
            torch::randint(0, base.size(dim), {index_size},
                           torch::TensorOptions(index_scalar_type));
        std::vector<int64_t> value_sizes(base.sizes().begin(),
                                         base.sizes().end());
        int canonical_dim = dim < 0 ? dim + rank : dim;
        value_sizes[canonical_dim] = index_size;
        torch::Tensor value =
            isFloatingType(scalar_type)
                ? torch::rand(value_sizes, torch::TensorOptions(scalar_type))
                : torch::randint(100, value_sizes,
                                 torch::TensorOptions(scalar_type));
        torch::Tensor result = torch::index_add(base, dim, index, value);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_base = CopyToDevice(base, device);
          torch::Tensor xla_index = CopyToDevice(index, device);
          torch::Tensor xla_value = CopyToDevice(value, device);
          torch::Tensor xla_result =
              torch::index_add(xla_base, dim, xla_index, xla_value);
          AllClose(result, xla_result);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::index_add", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestIndexAddInPlace) {
  int index_size = 10;
  int rank = 3;
  std::vector<double> alphas{0.0, 1.0, 2.0};

  for (torch::ScalarType scalar_type :
       {torch::kByte, torch::kFloat, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    for (int dim = -rank; dim < rank; ++dim) {
      for (double alpha : alphas) {
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor base =
              isFloatingType(scalar_type)
                  ? torch::rand({5, 3, 7}, torch::TensorOptions(scalar_type))
                  : torch::randint(50, {5, 3, 7},
                                   torch::TensorOptions(scalar_type));
          torch::Tensor index =
              torch::randint(0, base.size(dim), {index_size},
                             torch::TensorOptions(torch::kLong));
          std::vector<int64_t> value_sizes(base.sizes().begin(),
                                           base.sizes().end());
          int canonical_dim = dim < 0 ? dim + rank : dim;
          value_sizes[canonical_dim] = index_size;
          torch::Tensor value =
              isFloatingType(scalar_type)
                  ? torch::rand(value_sizes, torch::TensorOptions(scalar_type))
                  : torch::randint(50, value_sizes,
                                   torch::TensorOptions(scalar_type));
          torch::Tensor xla_base = CopyToDevice(base.clone(), device);
          torch::Tensor xla_index = CopyToDevice(index, device);
          torch::Tensor xla_value = CopyToDevice(value, device);
          torch::Tensor xla_result =
              xla_base.index_add_(dim, xla_index, xla_value, alpha);
          torch::Tensor result = base.index_add_(dim, index, value, alpha);
          AllClose(result, xla_result);
          AllClose(base, xla_base);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::index_add", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestIndexAddRank0) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor base =
        isFloatingType(scalar_type)
            ? torch::rand({5, 3, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {5, 3, 7}, torch::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor index = torch::randint(0, base.size(dim), at::IntArrayRef{},
                                           torch::TensorOptions(torch::kLong));
      std::vector<int64_t> value_sizes(base.sizes().begin(),
                                       base.sizes().end());
      int canonical_dim = dim < 0 ? dim + rank : dim;
      value_sizes[canonical_dim] = 1;
      torch::Tensor value =
          isFloatingType(scalar_type)
              ? torch::rand(value_sizes, torch::TensorOptions(scalar_type))
              : torch::randint(100, value_sizes,
                               torch::TensorOptions(scalar_type));
      torch::Tensor result = torch::index_add(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_base = CopyToDevice(base, device);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_value = CopyToDevice(value, device);
        torch::Tensor xla_result =
            torch::index_add(xla_base, dim, xla_index, xla_value);
        AllEqual(result, xla_result);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_add", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexCopy) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor base =
        isFloatingType(scalar_type)
            ? torch::rand({5, 3, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {5, 3, 7}, torch::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor index =
          torch::randperm(base.size(dim), torch::TensorOptions(torch::kLong));
      torch::Tensor value =
          isFloatingType(scalar_type)
              ? torch::rand(base.sizes(), torch::TensorOptions(scalar_type))
              : torch::randint(100, base.sizes(),
                               torch::TensorOptions(scalar_type));
      torch::Tensor result = torch::index_copy(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_base = CopyToDevice(base, device);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_value = CopyToDevice(value, device);
        torch::Tensor xla_result =
            torch::index_copy(xla_base, dim, xla_index, xla_value);
        AllEqual(result, xla_result);
      });
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::index_copy", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestIndexCopyInPlace) {
  int index_size = 10;
  int rank = 3;
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice(
          {XlaDeviceType::CPU, XlaDeviceType::TPU},
          [&](const torch::Device& device) {
            torch::Tensor base =
                isFloatingType(scalar_type)
                    ? torch::rand({5, 3, 7}, torch::TensorOptions(scalar_type))
                    : torch::randint(100, {5, 3, 7},
                                     torch::TensorOptions(scalar_type));
            torch::Tensor index =
                torch::randint(0, base.size(dim), {index_size},
                               torch::TensorOptions(torch::kLong));
            std::vector<int64_t> value_sizes(base.sizes().begin(),
                                             base.sizes().end());
            int canonical_dim = dim < 0 ? dim + rank : dim;
            value_sizes[canonical_dim] = index_size;
            torch::Tensor value =
                isFloatingType(scalar_type)
                    ? torch::rand(value_sizes,
                                  torch::TensorOptions(scalar_type))
                    : torch::randint(100, value_sizes,
                                     torch::TensorOptions(scalar_type));
            torch::Tensor xla_base = CopyToDevice(base.clone(), device);
            torch::Tensor result = base.index_copy(dim, index, value);
            torch::Tensor xla_index = CopyToDevice(index, device);
            torch::Tensor xla_value = CopyToDevice(value, device);
            torch::Tensor xla_result =
                xla_base.index_copy(dim, xla_index, xla_value);
            AllEqual(result, xla_result);
            AllEqual(base, xla_base);

            ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
            ExpectCounterChanged("xla::index_copy",
                                 cpp_test::GetIgnoredCounters());
          });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestIndexCopyRank0) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor base =
        isFloatingType(scalar_type)
            ? torch::rand({5, 3, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {5, 3, 7}, torch::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor index = torch::randint(0, base.size(dim), at::IntArrayRef{},
                                           torch::TensorOptions(torch::kLong));
      std::vector<int64_t> value_sizes(base.sizes().begin(),
                                       base.sizes().end());
      int canonical_dim = dim < 0 ? dim + rank : dim;
      value_sizes[canonical_dim] = 1;
      torch::Tensor value =
          isFloatingType(scalar_type)
              ? torch::rand(value_sizes, torch::TensorOptions(scalar_type))
              : torch::randint(100, value_sizes,
                               torch::TensorOptions(scalar_type));
      torch::Tensor result = torch::index_copy(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_base = CopyToDevice(base, device);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_value = CopyToDevice(value, device);
        torch::Tensor xla_result =
            torch::index_copy(xla_base, dim, xla_index, xla_value);
        AllEqual(result, xla_result);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::index_copy", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestRelu) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::relu(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::relu(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestReluInPlace) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::relu_(input);
    torch::Tensor xla_output = torch::relu_(xla_input);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestPrelu) {
  int channel_size = 3;
  torch::Tensor input =
      torch::rand({2, channel_size, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand(channel_size, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::prelu(input, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_weight = CopyToDevice(weight, device);
    torch::Tensor xla_output = torch::prelu(xla_input, xla_weight);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_prelu_kernel", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestPreluBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::prelu(inputs[0], inputs[1]);
  };
  torch::Tensor input = torch::rand(
      {5, 3}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  torch::Tensor weight = torch::rand({3}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    TestBackward({input, weight}, device, testfn);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_prelu_kernel_backward",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestHardshrink) {
  torch::Tensor input = torch::randn({10}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::hardshrink(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::hardshrink(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestHardshrinkWithMixedDataType) {
  if (UsingTpu()) {
    GTEST_SKIP();
  }
  torch::Tensor lambdaTensor =
      torch::scalar_tensor(0., torch::TensorOptions(torch::kFloat32));
  // It seems the below .item() will convert a kFloat64 to a kFloat32 if I
  // make the scalar tensor a kFloat32 type.
  torch::Scalar lambda = lambdaTensor.item();
  torch::Tensor input =
      torch::randn({10}, torch::TensorOptions(torch::kFloat64));

  torch::Tensor output = torch::hardshrink(input, lambda);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::hardshrink(xla_input, lambda);
    AllClose(output, xla_output);
  });
}

// Unlike Softshrink, a negative lambda is a valid input for Hardshrink.
TEST_F(AtenXlaTensorTest, TestHardshrinkWithNegativeLambda) {
  torch::Tensor input = torch::randn({10}, torch::TensorOptions(torch::kFloat));
  torch::Scalar lambd = -0.5;
  torch::Tensor output = torch::hardshrink(input, lambd);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::hardshrink(xla_input, lambd);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestHardSigmoid) {
  torch::Tensor input = torch::randn({10}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::hardsigmoid(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::hardsigmoid(xla_input);
    AllClose(output, xla_output);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::hardsigmoid", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestHardSigmoidInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor input =
        torch::randn({10}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::hardsigmoid_(input);
    torch::Tensor xla_output = torch::hardsigmoid_(xla_input);
    AllClose(input, xla_input);
    AllClose(output, xla_output);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::hardsigmoid", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestHardSigmoidBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::hardsigmoid(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::randn({10},
                      torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::hardsigmoid_backward",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestHardSwish) {
  torch::Tensor input = torch::randn({10}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::hardswish(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::hardswish(xla_input);
    AllClose(output, xla_output);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::hardswish", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestHardSwishInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor input =
        torch::randn({10}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::hardswish_(input);
    torch::Tensor xla_output = torch::hardswish_(xla_input);
    AllClose(input, xla_input);
    AllClose(output, xla_output);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::hardswish", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestHardSwishBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::hardswish(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::randn({10},
                      torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::hardswish_backward",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSoftshrink) {
  torch::Tensor input = torch::randn({10}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::softshrink(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::softshrink(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestHardtanh) {
  torch::Tensor input = torch::randn({10}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::hardtanh(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::hardtanh(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestHardtanhInPlace) {
  torch::Tensor input = torch::randn({10}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::hardtanh_(input);
    torch::Tensor xla_output = torch::hardtanh_(xla_input);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestLeakyRelu) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  double negative_slope = 0.01;
  torch::Tensor output = torch::leaky_relu(input, negative_slope);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::leaky_relu(xla_input, negative_slope);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestLeakyReluInPlace) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  double negative_slope = 0.01;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::leaky_relu_(input, negative_slope);
    torch::Tensor xla_output = torch::leaky_relu_(xla_input, negative_slope);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestExp) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::exp(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::exp(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestExpm1) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::expm1(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::expm1(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestLog) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::log(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::log(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestLog2) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::log2(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::log2(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestLog10) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::log10(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::log10(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestLog1p) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::log1p(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::log1p(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestErf) {
  torch::Tensor a = torch::randn({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::erf(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::erf(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestErfc) {
  torch::Tensor a = torch::randn({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::erfc(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::erfc(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestErfinv) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::erfinv(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::erfinv(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestSqrt) {
  torch::Tensor a =
      torch::abs(torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)));
  torch::Tensor b = torch::sqrt(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sqrt(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestRsqrt) {
  torch::Tensor a =
      torch::abs(torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)));
  torch::Tensor b = torch::rsqrt(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::rsqrt(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestReciprocal) {
  torch::Tensor a = torch::randn({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::reciprocal(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::reciprocal(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestPowTensorScalar) {
  torch::Tensor base = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar exponent = 4.09;
  torch::Tensor result = torch::pow(base, exponent);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_base = CopyToDevice(base, device);
    torch::Tensor xla_result = torch::pow(xla_base, exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestPowTensorScalarInPlace) {
  torch::Tensor base = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar exponent = 4.09;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_base = CopyToDevice(base.clone(), device);
    torch::Tensor result = base.pow_(exponent);
    torch::Tensor xla_result = xla_base.pow_(exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(base, xla_base, /*rtol=*/1e-3, /*atol=*/1e-5);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestPowTensorTensor) {
  torch::Tensor base =
      torch::abs(torch::rand({4, 2}, torch::TensorOptions(torch::kFloat)));
  torch::Tensor exponent = torch::rand({4, 2});
  torch::Tensor result = torch::pow(base, exponent);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_base = CopyToDevice(base, device);
    torch::Tensor xla_exponent = CopyToDevice(exponent, device);
    torch::Tensor xla_result = torch::pow(xla_base, xla_exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestPowTensorTensorInPlace) {
  torch::Tensor base =
      torch::abs(torch::rand({4, 2}, torch::TensorOptions(torch::kFloat)));
  torch::Tensor exponent = torch::rand({4, 2});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_base = CopyToDevice(base.clone(), device);
    torch::Tensor result = base.pow_(exponent);
    torch::Tensor xla_exponent = CopyToDevice(exponent, device);
    torch::Tensor xla_result = xla_base.pow_(xla_exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(base, xla_base, /*rtol=*/1e-3, /*atol=*/1e-5);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestPowTensorTensorBroadcast) {
  torch::Tensor base =
      torch::abs(torch::rand({4, 2}, torch::TensorOptions(torch::kFloat)));
  torch::Tensor exponent = torch::rand({4, 1});
  torch::Tensor result = torch::pow(base, exponent);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_base = CopyToDevice(base, device);
    torch::Tensor xla_exponent = CopyToDevice(exponent, device);
    torch::Tensor xla_result = torch::pow(xla_base, xla_exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestPowScalarTensor) {
  torch::Scalar base = 3.5;
  torch::Tensor exponent = torch::rand({4, 2});
  torch::Tensor result = torch::pow(base, exponent);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_exponent = CopyToDevice(exponent, device);
    torch::Tensor xla_result = torch::pow(base, xla_exponent).to(torch::kFloat);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestPowIntExponent) {
  torch::Tensor base =
      torch::abs(torch::rand({4, 2}, torch::TensorOptions(torch::kFloat)));
  torch::Scalar exponent = 3;
  torch::Tensor result = torch::pow(base, exponent);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_base = CopyToDevice(base, device);
    torch::Tensor xla_result = torch::pow(xla_base, exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestPowFloatScalarBaseIntExponent) {
  torch::Scalar base = .5;
  torch::Tensor exponent = torch::randint(0, 100, {4, 2});
  torch::Tensor result = torch::pow(base, exponent);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_exponent = CopyToDevice(exponent, device);
    torch::Tensor xla_result = torch::pow(base, xla_exponent).to(torch::kFloat);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestFmodScalar) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Scalar divisor = 2.0;
  torch::Tensor b = torch::fmod(a, divisor);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::fmod(xla_a, divisor);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestFmodScalarInPlace) {
  torch::Scalar divisor = 2.0;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a =
        torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = a.fmod_(divisor);
    torch::Tensor xla_b = xla_a.fmod_(divisor);
    AllClose(b, xla_b);
    AllClose(a, xla_a);
  });
}

TEST_F(AtenXlaTensorTest, TestFmodTensor) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 10.0;
  torch::Tensor c = torch::fmod(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::fmod(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestFmodTensorInPlace) {
  torch::Tensor b =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 10.0;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a =
        torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor c = a.fmod_(b);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = xla_a.fmod_(xla_b);
    AllClose(c, xla_c);
    AllClose(a, xla_a);
  });
}

TEST_F(AtenXlaTensorTest, TestRemainderScalar) {
  torch::Tensor a =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Scalar divisor = -2.0;
  torch::Tensor b = torch::remainder(a, divisor);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::remainder(xla_a, divisor);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestRemainderScalarInPlace) {
  torch::Scalar divisor = -2.0;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a =
        torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = a.remainder_(divisor);
    torch::Tensor xla_b = xla_a.remainder_(divisor);
    AllClose(b, xla_b);
    AllClose(a, xla_a);
  });
}

TEST_F(AtenXlaTensorTest, TestRemainderTensor) {
  torch::Tensor a =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 10.0;
  torch::Tensor c = torch::remainder(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::remainder(xla_a, xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-4, /*atol=*/1e-6);
  });
}

TEST_F(AtenXlaTensorTest, TestRemainderTensorInPlace) {
  torch::Tensor b =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 10.0;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a =
        torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor c = a.remainder_(b);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = xla_a.remainder_(xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-4, /*atol=*/1e-6);
    AllClose(a, xla_a, /*rtol=*/1e-4, /*atol=*/1e-6);
  });
}

}  // namespace cpp_test
}  // namespace torch_xla
