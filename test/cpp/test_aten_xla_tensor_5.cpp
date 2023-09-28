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

TEST_F(AtenXlaTensorTest, TestWhere) {
  torch::Tensor a = torch::rand({3, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 3}, torch::TensorOptions(torch::kByte));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = i == j;
    }
  }
  torch::Tensor d = torch::where(c, a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::where(xla_c, xla_a, xla_b);
    AllClose(d, xla_d);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::where", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestWhereBroadcast) {
  torch::Tensor a = torch::rand({3, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::zeros({}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 3}, torch::TensorOptions(torch::kByte));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = i == j;
    }
  }
  torch::Tensor d = torch::where(c, a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::where(xla_c, xla_a, xla_b);
    AllClose(d, xla_d);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::where", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestWhereAutograd) {
  torch::Tensor a = torch::rand({3, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 3}, torch::TensorOptions(torch::kByte));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = i == j;
    }
  }
  torch::Tensor d = torch::where(c, a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::where(xla_c, xla_a, xla_b);
    AllClose(d, xla_d);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::where", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestThreshold) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  float threshold = 0.4;
  float value = 20;
  torch::Tensor output = torch::threshold(input, threshold, value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::threshold(xla_input, threshold, value);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestThresholdInPlace) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = input.clone();
  float threshold = 0.4;
  float value = 20;
  torch::threshold_(output, threshold, value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_output = CopyToDevice(input, device);
    torch::threshold_(xla_output, threshold, value);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestElu) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Scalar alpha = 0.5;
  torch::Scalar scale = 2.5;
  torch::Scalar input_scale = 1.5;
  torch::Tensor output = torch::elu(input, alpha, scale, input_scale);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::elu(xla_input, alpha, scale, input_scale);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestEluInPlace) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Scalar alpha = 0.5;
  torch::Scalar scale = 2.5;
  torch::Scalar input_scale = 1.5;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::elu_(input, alpha, scale, input_scale);
    torch::Tensor xla_output =
        torch::elu_(xla_input, alpha, scale, input_scale);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestSelu) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::selu(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::selu(xla_input);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::elu", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSeluInPlace) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::selu_(input);
    torch::Tensor xla_output = torch::selu_(xla_input);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::elu", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestCelu) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Scalar alpha = 2.5;
  torch::Tensor output = torch::celu(input, alpha);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::celu(xla_input, alpha);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::celu", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestCeluInPlace) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Scalar alpha = 2.5;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::celu_(input, alpha);
    torch::Tensor xla_output = torch::celu_(xla_input, alpha);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::celu", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestGelu) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  for (const auto& approximate : {"none", "tanh"}) {
    torch::Tensor output = torch::gelu(input, approximate);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::gelu(xla_input, approximate);
      AllClose(output, xla_output);
    });
    ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::gelu", cpp_test::GetIgnoredCounters());
  }
}

TEST_F(AtenXlaTensorTest, TestAddMatMul) {
  int in_channels = 32;
  int out_channels = 320;
  int labels = 50;
  torch::Tensor input = torch::rand({in_channels, out_channels},
                                    torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({out_channels, labels}, torch::TensorOptions(torch::kFloat));
  torch::Tensor bias =
      torch::rand({labels}, torch::TensorOptions(torch::kFloat));
  // Test beta != 1. through the CPU interop.
  for (double beta : {1., 2.}) {
    torch::Tensor output = torch::addmm(bias, input, weight, /*beta=*/beta);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_weight = CopyToDevice(weight, device);
      torch::Tensor xla_bias = CopyToDevice(bias, device);
      torch::Tensor xla_output =
          torch::addmm(xla_bias, xla_input, xla_weight, /*beta=*/beta);
      AllClose(output, xla_output);
    });

    ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  }
}

TEST_F(AtenXlaTensorTest, TestEmbedding) {
  torch::Tensor a = torch::rand({32, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor i =
      torch::randint(0, 31, {3, 4}, torch::TensorOptions(torch::kLong));
  torch::Tensor b =
      torch::embedding(a, i, /*padding_idx=*/0, /*scale_grad_by_freq=*/false,
                       /*sparse=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_i = CopyToDevice(i, device);
    torch::Tensor xla_b = torch::embedding(xla_a, xla_i, /*padding_idx=*/0,
                                           /*scale_grad_by_freq=*/false,
                                           /*sparse=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestOneHot) {
  int num_classes = 5;
  torch::Tensor input =
      torch::randint(0, num_classes, {10}, torch::TensorOptions(torch::kLong));
  torch::Tensor output = torch::one_hot(input, num_classes);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::one_hot(xla_input, num_classes);
    AllEqual(output, xla_output);
  });

  // TODO: PT one_hot impl employs item() which could be eliminated.
  ExpectCounterNotChanged("aten::(?!_local_scalar_dense).*",
                          cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::scatter", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestTranspose) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::t(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::t(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestTransposeInPlace) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = input.t_();
    torch::Tensor xla_output = xla_input.t_();
    EXPECT_EQ(xla_output.sizes(), output.sizes());
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestReshape) {
  torch::Tensor input =
      torch::rand({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::reshape(input, {-1, 320});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::reshape(xla_input, {-1, 320});
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::view_copy", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestResize) {
  // Testing a resize_() with target size bigger than original size is not
  // possible, as we fill with zeros, while pytorch fills with random garbage.
  torch::Tensor input =
      torch::rand({2, 2, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor saved_input = input.clone();
  input.resize_({3, 3});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(saved_input, device);
    xla_input.resize_({3, 3});
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestViewResize) {
  torch::Tensor input =
      torch::zeros({8, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor saved_input = input.clone();
  torch::Tensor output = input.view({4, 4});
  output.resize_({3, 3});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(saved_input, device);
    torch::Tensor xla_output = xla_input.view({4, 4});
    xla_output.resize_({3, 3});
    AllClose(input, xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestView) {
  torch::Tensor input =
      torch::rand({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = input.view({-1, 320});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = xla_input.view({-1, 320});
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestViewMod) {
  torch::Tensor input =
      torch::zeros({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor one = torch::tensor(1.0, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = input.view({-1, 320});
  output.add_(one, 1.0);
  input.add_(one, 1.0);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xinput =
        torch::zeros({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(xinput, device);
    torch::Tensor xla_one = CopyToDevice(one, device);
    torch::Tensor xla_output = xla_input.view({-1, 320});
    xla_output.add_(xla_one, 1.0);
    xla_input.add_(xla_one, 1.0);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestViewModComplex) {
  torch::Tensor input =
      torch::zeros({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor one = torch::tensor(1.0, torch::TensorOptions(torch::kFloat));
  torch::Tensor output1 = input.view({-1, 320});
  output1.add_(one, 1.0);
  torch::Tensor output2 = input.view({-1, 160});
  output2.add_(one, 1.0);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xinput =
        torch::zeros({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(xinput, device);
    torch::Tensor xla_one = CopyToDevice(one, device);
    torch::Tensor xla_output1 = xla_input.view({-1, 320});
    xla_output1.add_(xla_one, 1.0);
    torch::Tensor xla_output2 = xla_input.view({-1, 160});
    xla_output2.add_(xla_one, 1.0);
    AllClose(output1, xla_output1);
    AllClose(output2, xla_output2);
  });
}

TEST_F(AtenXlaTensorTest, TestViewAsComplexCopy) {
  torch::Tensor input =
      torch::rand({5, 4, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::view_as_complex_copy(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::view_as_complex_copy(xla_input);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::view_as_complex_copy",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestViewAsRealCopy) {
  torch::Tensor input =
      torch::rand({5, 4, 2}, torch::TensorOptions(torch::kComplexFloat));
  torch::Tensor output = torch::view_as_real_copy(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::view_as_real_copy(xla_input);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::view_as_real_copy",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestViewOfViewMod) {
  torch::Tensor input =
      torch::zeros({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor one = torch::tensor(1.0, torch::TensorOptions(torch::kFloat));
  torch::Tensor output1 = input.view({-1, 320});
  output1.add_(one, 1.0);
  torch::Tensor output2 = output1.view({-1, 160});
  output2.add_(one, 1.0);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xinput =
        torch::zeros({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(xinput, device);
    torch::Tensor xla_one = CopyToDevice(one, device);
    torch::Tensor xla_output1 = xla_input.view({-1, 320});
    xla_output1.add_(xla_one, 1.0);
    torch::Tensor xla_output2 = xla_output1.view({-1, 160});
    xla_output2.add_(xla_one, 1.0);
    AllClose(output1, xla_output1);
    AllClose(output2, xla_output2);
  });
}

TEST_F(AtenXlaTensorTest, TestViewSqueezeAddInPlace) {
  torch::Tensor input =
      torch::zeros({2, 3, 1}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> view_size = {2, 3, 1, 1};
  int squeeze_dim = 2;
  torch::Tensor one = torch::tensor(1.0, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = input.view(view_size);
    output.squeeze_(squeeze_dim);
    output.add_(one, 1.0);
    torch::Tensor xla_one = CopyToDevice(one, device);
    torch::Tensor xla_output = xla_input.view(view_size);
    xla_output.squeeze_(squeeze_dim);
    xla_output.add_(xla_one, 1.0);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestUnsafeView) {
  torch::Tensor input =
      torch::rand({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::_unsafe_view(input, {-1, 320});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::_unsafe_view(xla_input, {-1, 320});
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestNarrow) {
  torch::Tensor a =
      torch::rand({8, 10, 4, 4}, torch::TensorOptions(torch::kFloat));
  for (int64_t dim : {1, -3}) {
    for (int64_t start : {2, -8}) {
      torch::Tensor b = a.narrow(dim, start, 6);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::Tensor xla_b = xla_a.narrow(dim, start, 6);
        AllClose(b, xla_b);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::slice_copy", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestNarrowUpdate) {
  for (int64_t dim : {1, -2}) {
    for (int64_t start : {2, -6}) {
      torch::Tensor a =
          torch::rand({3, 8, 3}, torch::TensorOptions(torch::kFloat));
      torch::Tensor a_copy = a.clone();
      torch::Tensor b =
          torch::rand({3, 4, 3}, torch::TensorOptions(torch::kFloat));
      torch::Tensor c = a.narrow(dim, start, 4);
      c.add_(b, 1.0);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a_copy, device);
        torch::Tensor xla_b = CopyToDevice(b, device);
        torch::Tensor xla_c = xla_a.narrow(dim, start, 4);
        xla_c.add_(xla_b, 1.0);
        AllClose(c, xla_c);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::slice_copy", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestNarrowUpdateBaseCheck) {
  for (int64_t dim : {0, -2}) {
    for (int64_t start : {2, -6}) {
      torch::Tensor a =
          torch::zeros({8, 3}, torch::TensorOptions(torch::kFloat));
      torch::Tensor a_copy = a.clone();
      torch::Tensor b =
          torch::ones({4, 3}, torch::TensorOptions(torch::kFloat));
      torch::Tensor c = a.narrow(dim, start, 4);
      c.add_(b, 1.0);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a_copy, device);
        torch::Tensor xla_b = CopyToDevice(b, device);
        torch::Tensor xla_c = xla_a.narrow(dim, start, 4);
        xla_c.add_(xla_b, 1.0);
        AllClose(a, xla_a);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::slice_copy", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestNarrowUpdateTwoSlices) {
  for (int64_t dim : {0, -2}) {
    for (int64_t start0 : {2, -6}) {
      for (int64_t start1 : {6, -2}) {
        torch::Tensor a =
            torch::zeros({8, 3}, torch::TensorOptions(torch::kFloat));
        torch::Tensor a_copy = a.clone();
        torch::Tensor b =
            torch::ones({2, 3}, torch::TensorOptions(torch::kFloat));
        torch::Tensor c = b + 1;
        torch::Tensor d = a.narrow(dim, start0, 2);
        torch::Tensor e = a.narrow(dim, start1, 2);
        d.add_(b, 1.0);
        e.add_(c, 1.0);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a_copy, device);
          torch::Tensor xla_b = CopyToDevice(b, device);
          torch::Tensor xla_c = CopyToDevice(c, device);
          torch::Tensor xla_d = xla_a.narrow(dim, start0, 2);
          torch::Tensor xla_e = xla_a.narrow(dim, start1, 2);
          xla_d.add_(xla_b, 1.0);
          xla_e.add_(xla_c, 1.0);
          AllClose(d, xla_d);
          AllClose(e, xla_e);
          AllClose(a, xla_a);
        });

        ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
        ExpectCounterChanged("xla::slice_copy", cpp_test::GetIgnoredCounters());
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestNarrowUpdateView) {
  for (int64_t dim : {0, -3}) {
    for (int64_t start : {2, -6}) {
      torch::Tensor a =
          torch::rand({8, 2, 3}, torch::TensorOptions(torch::kFloat));
      torch::Tensor a_copy = a.clone();
      torch::Tensor b =
          torch::rand({4, 6}, torch::TensorOptions(torch::kFloat));
      torch::Tensor c = a.narrow(dim, start, 4);
      torch::Tensor d = c.view({4, 6});
      d.add_(b, 1.0);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a_copy, device);
        torch::Tensor xla_b = CopyToDevice(b, device);
        torch::Tensor xla_c = xla_a.narrow(dim, start, 4);
        torch::Tensor xla_d = xla_c.view({4, 6});
        xla_d.add_(xla_b, 1.0);
        AllClose(d, xla_d);
      });

      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
      ExpectCounterChanged("xla::slice_copy", cpp_test::GetIgnoredCounters());
    }
  }
}

TEST_F(AtenXlaTensorTest, TestNarrowInNarrowUpdate) {
  for (int64_t dim : {1, -2}) {
    for (int64_t start0 : {1, -7}) {
      for (int64_t start1 : {1, -5}) {
        torch::Tensor a =
            torch::rand({3, 8, 3}, torch::TensorOptions(torch::kFloat));
        torch::Tensor a_copy = a.clone();
        torch::Tensor b =
            torch::rand({3, 2, 3}, torch::TensorOptions(torch::kFloat));
        torch::Tensor c = a.narrow(dim, start0, 6);
        torch::Tensor d = c.narrow(dim, start1, 2);
        d.add_(b, 1.0);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a_copy, device);
          torch::Tensor xla_b = CopyToDevice(b, device);
          torch::Tensor xla_c = xla_a.narrow(dim, start0, 6);
          torch::Tensor xla_d = xla_c.narrow(dim, start1, 2);
          xla_d.add_(xla_b, 1.0);
          AllClose(a, xla_a);
        });

        ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
        ExpectCounterChanged("xla::slice_copy", cpp_test::GetIgnoredCounters());
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestNarrowCopy) {
  for (int64_t dim : {1, -3}) {
    for (int64_t start : {2, -8}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor input =
            torch::rand({8, 10, 4, 4}, torch::TensorOptions(torch::kFloat));
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor result = input.narrow_copy(dim, start, 6);
        input.add_(1);
        torch::Tensor xla_result = xla_input.narrow_copy(dim, start, 6);
        xla_input.add_(1);
        AllClose(result, xla_result);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestViewAs) {
  torch::Tensor input =
      torch::rand({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor empty = torch::empty({32, 320});
  torch::Tensor output = input.view_as(empty);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_empty = CopyToDevice(empty, device);
    torch::Tensor xla_output = xla_input.view_as(xla_empty);
    AllClose(output, xla_output);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::view_copy", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLogSoftmax) {
  torch::Tensor input =
      torch::rand({5, 3, 4, 2}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output = torch::log_softmax(input, dim);
      torch::Tensor xla_output = torch::log_softmax(xla_input, dim);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_log_softmax", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLogSoftmaxCast) {
  torch::Tensor input =
      torch::rand({5, 3, 4, 2}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output = torch::log_softmax(input, dim, torch::kDouble);
      torch::Tensor xla_output =
          torch::log_softmax(xla_input, dim, torch::kDouble);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_log_softmax", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSoftmax) {
  torch::Tensor input =
      torch::rand({10, 8, 24, 16}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output = torch::softmax(input, dim);
      torch::Tensor xla_output = torch::softmax(xla_input, dim);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_softmax", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSoftmaxCast) {
  torch::Tensor input =
      torch::rand({10, 8, 24, 16}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output = torch::softmax(input, dim, torch::kDouble);
      torch::Tensor xla_output = torch::softmax(xla_input, dim, torch::kDouble);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_softmax", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSoftmaxWrapper) {
  torch::Tensor input =
      torch::rand({10, 8, 24, 16}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output =
          torch::_softmax(input, dim, /*half_to_float=*/false);
      torch::Tensor xla_output =
          torch::_softmax(xla_input, dim, /*half_to_float=*/false);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_softmax", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSoftplus) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::softplus(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::softplus(xla_input);
    AllClose(output, xla_output, /*rtol=*/1e-4);
  });
}

TEST_F(AtenXlaTensorTest, TestMaxPool1D) {
  torch::Tensor input =
      torch::rand({1, 64, 112}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output =
              torch::max_pool1d(input, /*kernel_size=*/{kernel_size},
                                /*stride=*/{stride},
                                /*padding=*/{padding}, /*dilation=*/{dilation},
                                /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output =
                torch::max_pool1d(xla_input,
                                  /*kernel_size=*/{kernel_size},
                                  /*stride=*/{stride},
                                  /*padding=*/{padding},
                                  /*dilation=*/{dilation},
                                  /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });

          ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
          ExpectCounterChanged("xla::max_pool2d_with_indices",
                               cpp_test::GetIgnoredCounters());
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool2D) {
  torch::Tensor input =
      torch::rand({1, 64, 112, 112}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output =
                torch::max_pool2d(xla_input,
                                  /*kernel_size=*/{kernel_size, kernel_size},
                                  /*stride=*/{stride, stride},
                                  /*padding=*/{padding, padding},
                                  /*dilation=*/{dilation, dilation},
                                  /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });

          ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
          ExpectCounterChanged("xla::max_pool2d",
                               cpp_test::GetIgnoredCounters());
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool2DWithIndices) {
  torch::Tensor input =
      torch::rand({1, 64, 112, 112}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          auto outputs = torch::max_pool2d_with_indices(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            auto xla_outputs = torch::max_pool2d_with_indices(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*dilation=*/{dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(std::get<0>(outputs), std::get<0>(xla_outputs));
            AllClose(std::get<1>(outputs), std::get<1>(xla_outputs));
          });

          ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
          ExpectCounterChanged("xla::max_pool2d_with_indices",
                               cpp_test::GetIgnoredCounters());
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool2DNonSquare) {
  torch::Tensor input =
      torch::rand({1, 64, 112, 112}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size + 1},
              /*stride=*/{stride, stride + 1},
              /*padding=*/{padding, padding + 1},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::max_pool2d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1},
                /*stride=*/{stride, stride + 1},
                /*padding=*/{padding, padding + 1},
                /*dilation=*/{dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });

          ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
          ExpectCounterChanged("xla::max_pool2d",
                               cpp_test::GetIgnoredCounters());
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool3D) {
  torch::Tensor input =
      torch::rand({1, 64, 16, 16, 16}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool3d(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::max_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });

          ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
          ExpectCounterChanged("xla::max_pool3d",
                               cpp_test::GetIgnoredCounters());
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool3DWithIndices) {
  torch::Tensor input =
      torch::rand({1, 64, 16, 16, 16}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          auto outputs = torch::max_pool3d_with_indices(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            auto xla_outputs = torch::max_pool3d_with_indices(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);

            AllClose(std::get<0>(outputs), std::get<0>(xla_outputs));
            AllClose(std::get<1>(outputs), std::get<1>(xla_outputs));
          });

          ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
          ExpectCounterChanged("xla::max_pool3d_with_indices",
                               cpp_test::GetIgnoredCounters());
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool3DIncompleteAttributes) {
  torch::Tensor input =
      torch::rand({1, 64, 16, 16, 16}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool3d(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{},
              /*padding=*/{padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::max_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{},
                /*padding=*/{padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });

          ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
          ExpectCounterChanged("xla::max_pool3d",
                               cpp_test::GetIgnoredCounters());
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool3DNonSquare) {
  torch::Tensor input =
      torch::rand({1, 64, 16, 16, 16}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
              /*stride=*/{stride, stride + 1, stride},
              /*padding=*/{padding, padding + 1, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::max_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
                /*stride=*/{stride, stride + 1, stride},
                /*padding=*/{padding, padding + 1, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });

          ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
          ExpectCounterChanged("xla::max_pool3d",
                               cpp_test::GetIgnoredCounters());
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool2DNoBatch) {
  torch::Tensor input =
      torch::rand({64, 112, 112}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output =
                torch::max_pool2d(xla_input,
                                  /*kernel_size=*/{kernel_size, kernel_size},
                                  /*stride=*/{stride, stride},
                                  /*padding=*/{padding, padding},
                                  /*dilation=*/{dilation, dilation},
                                  /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });

          ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
          ExpectCounterChanged("xla::max_pool2d",
                               cpp_test::GetIgnoredCounters());
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool3DNoBatch) {
  torch::Tensor input =
      torch::rand({64, 16, 16, 16}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool3d(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::max_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });

          ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
          ExpectCounterChanged("xla::max_pool3d",
                               cpp_test::GetIgnoredCounters());
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAvgPool1D) {
  torch::Tensor input =
      torch::rand({4, 1, 28}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output =
              torch::avg_pool1d(input, /*kernel_size=*/{kernel_size},
                                /*stride=*/{stride},
                                /*padding=*/{padding}, /*ceil_mode=*/ceil_mode,
                                /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output =
                torch::avg_pool1d(xla_input,
                                  /*kernel_size=*/{kernel_size},
                                  /*stride=*/{stride},
                                  /*padding=*/{padding},
                                  /*ceil_mode=*/ceil_mode,
                                  /*count_include_pad=*/count_include_pad);
            AllClose(output, xla_output);
          });

          ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
          ExpectCounterChanged("xla::avg_pool2d",
                               cpp_test::GetIgnoredCounters());
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAvgPool2D) {
  torch::Tensor input =
      torch::rand({4, 1, 28, 28}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            // torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output =
                torch::avg_pool2d(xla_input,
                                  /*kernel_size=*/{kernel_size, kernel_size},
                                  /*stride=*/{stride, stride},
                                  /*padding=*/{padding, padding},
                                  /*ceil_mode=*/ceil_mode,
                                  /*count_include_pad=*/count_include_pad);
            AllClose(output, xla_output.to(torch::kCPU));
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAvgPool2DNonSquare) {
  torch::Tensor input =
      torch::rand({4, 1, 28, 28}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size + 1},
              /*stride=*/{stride, stride + 1},
              /*padding=*/{padding, padding + 1}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::avg_pool2d(
                xla_input,
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
  torch::Tensor input =
      torch::rand({4, 1, 28, 28, 28}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool3d(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::avg_pool3d(
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
  torch::Tensor input =
      torch::rand({4, 1, 28, 28, 28}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool3d(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{},
              /*padding=*/{padding, padding, padding}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::avg_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{},
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

TEST_F(AtenXlaTensorTest, TestAvgPool3DNonSquare) {
  torch::Tensor input =
      torch::rand({4, 1, 28, 28, 28}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
              /*stride=*/{stride, stride + 1, stride},
              /*padding=*/{padding, padding + 1, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::avg_pool3d(
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

TEST_F(AtenXlaTensorTest, TestAvgPool2DNoBatch) {
  torch::Tensor input =
      torch::rand({1, 28, 28}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output =
                torch::avg_pool2d(xla_input,
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

TEST_F(AtenXlaTensorTest, TestAvgPool3DNoBatch) {
  torch::Tensor input =
      torch::rand({1, 28, 28, 28}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool3d(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::avg_pool3d(
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

TEST_F(AtenXlaTensorTest, TestAdaptiveMaxPool2D) {
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(bridge::GetDefaultDevice()->type());
  // skip this test until the tile mismatch bug is fixed.
  if (hw_type == XlaDeviceType::TPU) {
    return;
  }
  std::vector<torch::Tensor> inputs = {
      torch::rand({2, 10, 10}, torch::TensorOptions(torch::kFloat)),
      torch::rand({2, 2, 10, 10}, torch::TensorOptions(torch::kFloat)),
  };
  std::vector<std::vector<int64_t>> dim_sizes = {{2, 2}, {5, 2}, {5, 5}};

  for (torch::Tensor input : inputs) {
    for (auto output_size : dim_sizes) {
      std::tuple<at::Tensor, at::Tensor> output =
          torch::adaptive_max_pool2d(input, output_size);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        std::tuple<at::Tensor, at::Tensor> xla_output =
            torch::adaptive_max_pool2d(xla_input, output_size);
        AllClose(std::get<0>(output), std::get<0>(xla_output));
        AllClose(std::get<1>(output), std::get<1>(xla_output));
      });
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::adaptive_max_pool2d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAdaptiveMaxPool2DBackward) {
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(bridge::GetDefaultDevice()->type());
  // skip this test until the tile mismatch bug is fixed.
  if (hw_type == XlaDeviceType::TPU) {
    return;
  }
  std::vector<torch::Tensor> inputs = {
      torch::rand({2, 10, 10},
                  torch::TensorOptions(torch::kFloat).requires_grad(true)),
      torch::rand({2, 2, 10, 10},
                  torch::TensorOptions(torch::kFloat).requires_grad(true)),
  };
  std::vector<std::vector<int64_t>> dim_sizes = {{2, 2}, {5, 2}, {5, 5}};
  for (auto output_size : dim_sizes) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return std::get<0>(torch::adaptive_max_pool2d(inputs[0], output_size));
    };
    ForEachDevice([&](const torch::Device& device) {
      for (torch::Tensor input : inputs) {
        TestBackward(
            {torch::rand(
                {4, 1, 10, 10},
                torch::TensorOptions(torch::kFloat).requires_grad(true))},
            device, testfn);
      }
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::adaptive_max_pool2d_backward",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAdaptiveAvgPool2D) {
  torch::Tensor input =
      torch::rand({4, 1, 28, 28}, torch::TensorOptions(torch::kFloat));
  for (int64_t output_size : {7, 4}) {
    torch::Tensor output =
        torch::adaptive_avg_pool2d(input, {output_size, output_size});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output =
          torch::adaptive_avg_pool2d(xla_input, {output_size, output_size});
      AllClose(output, xla_output);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_adaptive_avg_pool2d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAdaptiveAvgPool3D) {
  torch::Tensor input =
      torch::rand({9, 4, 56, 28, 28}, torch::TensorOptions(torch::kFloat));
  for (int64_t output_size : {7, 4}) {
    torch::Tensor output = torch::adaptive_avg_pool3d(
        input, {output_size, output_size, output_size});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::adaptive_avg_pool3d(
          xla_input, {output_size, output_size, output_size});
      AllClose(output, xla_output);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_adaptive_avg_pool3d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAdaptiveAvgPool3DNoBatch) {
  torch::Tensor input =
      torch::rand({3, 56, 28, 28}, torch::TensorOptions(torch::kFloat));
  for (int64_t output_size : {7, 4}) {
    torch::Tensor output = torch::adaptive_avg_pool3d(
        input, {output_size, output_size, output_size});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::adaptive_avg_pool3d(
          xla_input, {output_size, output_size, output_size});
      AllClose(output, xla_output);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_adaptive_avg_pool3d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestAdaptiveAvgPool2DNoBatch) {
  torch::Tensor input =
      torch::rand({1, 56, 56}, torch::TensorOptions(torch::kFloat));
  for (int64_t output_size : {7, 8}) {
    torch::Tensor output =
        torch::adaptive_avg_pool2d(input, {output_size, output_size});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output =
          torch::adaptive_avg_pool2d(xla_input, {output_size, output_size});
      AllClose(output, xla_output);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_adaptive_avg_pool2d",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMaxUnpool2D) {
  int kernel_size = 2;
  torch::Tensor input =
      torch::rand({2, 2, 8, 8}, torch::TensorOptions(torch::kFloat));
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output;
          torch::Tensor indices;
          std::tie(output, indices) = torch::max_pool2d_with_indices(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          std::vector<int64_t> output_size({input.size(2), input.size(3)});
          at::Tensor utensor =
              torch::max_unpool2d(output, indices, output_size);

          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_output = CopyToDevice(output, device);
            torch::Tensor xla_indices = CopyToDevice(indices, device);
            at::Tensor xla_utensor =
                torch::max_unpool2d(xla_output, xla_indices, output_size);
            AllClose(utensor, xla_utensor);
          });
        }
      }
    }
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::max_unpool2d", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMaxUnpool3D) {
  int kernel_size = 2;
  torch::Tensor input =
      torch::rand({2, 2, 8, 8, 8}, torch::TensorOptions(torch::kFloat));
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output;
          torch::Tensor indices;
          std::tie(output, indices) = torch::max_pool3d_with_indices(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          std::vector<int64_t> output_size(
              {input.size(2), input.size(3), input.size(4)});
          at::Tensor utensor = torch::max_unpool3d(
              output, indices, output_size, /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding});

          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_output = CopyToDevice(output, device);
            torch::Tensor xla_indices = CopyToDevice(indices, device);
            at::Tensor xla_utensor =
                torch::max_unpool3d(xla_output, xla_indices, output_size,
                                    /*stride=*/{stride, stride, stride},
                                    /*padding=*/{padding, padding, padding});
            AllClose(utensor, xla_utensor);
          });
        }
      }
    }
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::max_unpool3d", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestNllLoss) {
  int batch = 6;
  int classes = 2;
  for (auto dtype : {torch::kFloat, torch::kDouble}) {
    for (int ignore_index : {-1, 0, 1, 5}) {
      for (bool def_weight : {false, true}) {
        torch::Tensor input =
            torch::rand({batch, classes}, torch::TensorOptions(dtype));
        torch::Tensor target =
            torch::randint(std::min(ignore_index, 0), classes, {batch},
                           torch::TensorOptions(torch::kLong));
        torch::Tensor weight;
        if (def_weight) {
          weight = torch::rand({classes}, torch::TensorOptions(dtype));
        }
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean, torch::Reduction::Sum,
              torch::Reduction::None}) {
          torch::Tensor output =
              torch::nll_loss(/*self=*/input, /*target=*/target,
                              /*weight=*/weight,
                              /*reduction=*/reduction,
                              /*ignore_index=*/ignore_index);

          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_target = CopyToDevice(target, device);
            torch::Tensor xla_weight =
                def_weight ? CopyToDevice(weight, device) : torch::Tensor();
            torch::Tensor xla_output = torch::nll_loss(
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
  ExpectCounterChanged("xla::nll_loss_forward", cpp_test::GetIgnoredCounters());
}

}  // namespace cpp_test
}  // namespace torch_xla
