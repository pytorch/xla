#include <gtest/gtest.h>

#include <vector>

#include <ATen/ATen.h>
#include "cpp_test_util.h"
#include "tensor.h"
#include "torch/csrc/autograd/variable.h"
#include "torch_xla_test.h"

namespace torch_xla {
namespace cpp_test {

using TensorTest = TorchXlaTest;

TEST_F(TensorTest, TestAdd) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  auto c = a.add(b, 1.0);

  ForEachDevice([&](const Device& device) {
    auto dev_a = XLATensor::Create(a, device, /*requires_grad=*/false);
    auto dev_b = XLATensor::Create(b, device, /*requires_grad=*/false);
    auto dev_c = dev_a->add(*dev_b, 1.0);

    AllClose(c, *dev_c);
  });
}

TEST_F(TensorTest, TestIntegerAdd) {
  std::vector<at::ScalarType> types(
      {at::kByte, at::kChar, at::kShort, at::kInt, at::kLong});

  ForEachDevice([&](const Device& device) {
    for (auto type : types) {
      at::Tensor a = at::randint(0, 63, {2, 2}, at::TensorOptions(type));
      at::Tensor b = at::randint(0, 63, {2, 2}, at::TensorOptions(type));
      auto c = a.add(b, 1.0);

      auto dev_a = XLATensor::Create(a, device, /*requires_grad=*/false);
      auto dev_b = XLATensor::Create(b, device, /*requires_grad=*/false);
      auto dev_c = dev_a->add(*dev_b, 1.0);

      EXPECT_TRUE(EqualValues(c, ToTensor(*dev_c)));
    }
  });
}

TEST_F(TensorTest, TestSize) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  ForEachDevice([&](const Device& device) {
    auto dev_input = XLATensor::Create(input, device, /*requires_grad=*/false);
    for (int dim = -rank; dim < rank; ++dim) {
      EXPECT_EQ(input.size(dim), dev_input->size(dim));
    }
  });
}

TEST_F(TensorTest, TestRelu) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  auto output = input.relu();
  ForEachDevice([&](const Device& device) {
    auto dev_input = XLATensor::Create(input, device, /*requires_grad=*/false);
    auto dev_output = dev_input->relu();
    AllClose(output, *dev_output);
  });
}

TEST_F(TensorTest, TestThreshold) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  float threshold = 0.4;
  float value = 20;
  auto output = at::threshold(input, threshold, value);
  ForEachDevice([&](const Device& device) {
    auto dev_input = XLATensor::Create(input, device, /*requires_grad=*/false);
    auto dev_output = dev_input->threshold(threshold, value);
    AllClose(output, *dev_output);
  });
}

TEST_F(TensorTest, TestAddMatMul) {
  int in_channels = 32;
  int out_channels = 320;
  int labels = 50;
  at::Tensor input =
      at::rand({in_channels, out_channels}, at::TensorOptions(at::kFloat));
  at::Tensor weight =
      at::rand({out_channels, labels}, at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({labels}, at::TensorOptions(at::kFloat));
  auto output = at::addmm(bias, input, weight);
  ForEachDevice([&](const Device& device) {
    auto dev_input = XLATensor::Create(input, device, /*requires_grad=*/false);
    auto dev_weight =
        XLATensor::Create(weight, device, /*requires_grad=*/false);
    auto dev_bias = XLATensor::Create(bias, device, /*requires_grad=*/false);
    auto dev_output = dev_input->addmm(*dev_weight, *dev_bias,
                                       /*use_full_conv_precision=*/true);
    AllClose(output, *dev_output);
  });
}

TEST_F(TensorTest, TestTranspose) {
  at::Tensor input = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  auto output = input.t();
  ForEachDevice([&](const Device& device) {
    auto dev_input = XLATensor::Create(input, device, /*requires_grad=*/false);
    auto dev_output = dev_input->t();
    AllClose(output, *dev_output);
  });
}

TEST_F(TensorTest, TestView) {
  at::Tensor input = at::rand({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  auto output = input.view({-1, 320});
  ForEachDevice([&](const Device& device) {
    auto dev_input = XLATensor::Create(input, device, /*requires_grad=*/false);
    auto dev_output = dev_input->view({-1, 320});
    AllClose(output, *dev_output);
  });
}

TEST_F(TensorTest, TestLogSoftmax) {
  at::Tensor input = at::rand({5, 3, 4, 2}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    auto dev_input = XLATensor::Create(input, device, /*requires_grad=*/false);
    for (int dim = 0; dim < input.dim(); ++dim) {
      auto output = input.log_softmax(dim);
      auto dev_output = dev_input->log_softmax(dim);
      AllClose(output, *dev_output);
    }
  });
}

TEST_F(TensorTest, TestMaxPool2D) {
  at::Tensor input = at::rand({1, 64, 112, 112}, at::TensorOptions(at::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      auto output =
          at::max_pool2d(input, /*kernel_size=*/{kernel_size, kernel_size},
                         /*stride=*/{stride, stride},
                         /*padding=*/{padding, padding}, /*dilation=*/{1, 1},
                         /*ceil_mode=*/false);
      ForEachDevice([&](const Device& device) {
        auto dev_input =
            XLATensor::Create(input, device, /*requires_grad=*/false);
        auto dev_output = dev_input->max_pool2d(
            /*kernel_size=*/{kernel_size, kernel_size},
            /*stride=*/{stride, stride},
            /*padding=*/{padding, padding});
        AllClose(output, *dev_output);
      });
    }
  }
}

TEST_F(TensorTest, TestMaxPool2DNonSquare) {
  at::Tensor input = at::rand({1, 64, 112, 112}, at::TensorOptions(at::kFloat));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      auto output = at::max_pool2d(
          input, /*kernel_size=*/{kernel_size, kernel_size + 1},
          /*stride=*/{stride, stride + 1},
          /*padding=*/{padding, padding + 1}, /*dilation=*/{1, 1},
          /*ceil_mode=*/false);
      ForEachDevice([&](const Device& device) {
        auto dev_input =
            XLATensor::Create(input, device, /*requires_grad=*/false);
        auto dev_output = dev_input->max_pool2d(
            /*kernel_size=*/{kernel_size, kernel_size + 1},
            /*stride=*/{stride, stride + 1},
            /*padding=*/{padding, padding + 1});
        AllClose(output, *dev_output);
      });
    }
  }
}

TEST_F(TensorTest, TestAvgPool2D) {
  at::Tensor input = at::rand({4, 1, 28, 28}, at::TensorOptions(at::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        auto output =
            at::avg_pool2d(input, /*kernel_size=*/{kernel_size, kernel_size},
                           /*stride=*/{stride, stride},
                           /*padding=*/{padding, padding}, /*ceil_mode=*/false,
                           count_include_pad);
        ForEachDevice([&](const Device& device) {
          auto dev_input =
              XLATensor::Create(input, device, /*requires_grad=*/false);
          auto dev_output = dev_input->avg_pool2d(
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, count_include_pad);
          AllClose(output, *dev_output);
        });
      }
    }
  }
}

TEST_F(TensorTest, TestAvgPool2DNonSquare) {
  at::Tensor input = at::rand({4, 1, 28, 28}, at::TensorOptions(at::kFloat));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        auto output = at::avg_pool2d(
            input, /*kernel_size=*/{kernel_size, kernel_size + 1},
            /*stride=*/{stride, stride + 1},
            /*padding=*/{padding, padding + 1}, /*ceil_mode=*/false,
            /*count_include_pad=*/count_include_pad);
        ForEachDevice([&](const Device& device) {
          auto dev_input =
              XLATensor::Create(input, device, /*requires_grad=*/false);
          auto dev_output = dev_input->avg_pool2d(
              /*kernel_size=*/{kernel_size, kernel_size + 1},
              /*stride=*/{stride, stride + 1},
              /*padding=*/{padding, padding + 1},
              /*count_include_pad=*/count_include_pad);
          AllClose(output, *dev_output);
        });
      }
    }
  }
}

TEST_F(TensorTest, TestConv2D) {
  int in_channels = 3;
  int out_channels = 7;
  int kernel_size = 5;
  at::Tensor input =
      at::rand({4, in_channels, 28, 28}, at::TensorOptions(at::kFloat));
  at::Tensor weight =
      at::rand({out_channels, in_channels, kernel_size, kernel_size},
               at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({out_channels}, at::TensorOptions(at::kFloat));
  at::Tensor no_bias;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        auto output =
            at::native::conv2d(input, weight, with_bias ? bias : no_bias,
                               /*stride=*/{stride, stride},
                               /*padding=*/{padding, padding});
        ForEachDevice([&](const Device& device) {
          auto dev_input =
              XLATensor::Create(input, device, /*requires_grad=*/false);
          auto dev_weight =
              XLATensor::Create(weight, device, /*requires_grad=*/false);
          std::shared_ptr<XLATensor> dev_output;
          if (with_bias) {
            auto dev_bias =
                XLATensor::Create(bias, device, /*requires_grad=*/false);
            dev_output = dev_input->conv2d(*dev_weight, *dev_bias,
                                           /*stride=*/{stride, stride},
                                           /*padding=*/{padding, padding},
                                           /*use_full_conv_precision=*/true);
          } else {
            dev_output = dev_input->conv2d(*dev_weight,
                                           /*stride=*/{stride, stride},
                                           /*padding=*/{padding, padding},
                                           /*use_full_conv_precision=*/true);
          }
          AllClose(output, *dev_output);
        });
      };
    }
  }
}

TEST_F(TensorTest, TestConv2DNonSquare) {
  int in_channels = 3;
  int out_channels = 7;
  int kernel_size = 5;
  at::Tensor input =
      at::rand({4, in_channels, 28, 28}, at::TensorOptions(at::kFloat));
  at::Tensor weight =
      at::rand({out_channels, in_channels, kernel_size, kernel_size},
               at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({out_channels}, at::TensorOptions(at::kFloat));
  at::Tensor no_bias;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        auto output =
            at::native::conv2d(input, weight, with_bias ? bias : no_bias,
                               /*stride=*/{stride, stride + 1},
                               /*padding=*/{padding, padding + 1});
        ForEachDevice([&](const Device& device) {
          auto dev_input =
              XLATensor::Create(input, device, /*requires_grad=*/false);
          auto dev_weight =
              XLATensor::Create(weight, device, /*requires_grad=*/false);
          std::shared_ptr<XLATensor> dev_output;
          if (with_bias) {
            auto dev_bias =
                XLATensor::Create(bias, device, /*requires_grad=*/false);
            dev_output = dev_input->conv2d(*dev_weight, *dev_bias,
                                           /*stride=*/{stride, stride + 1},
                                           /*padding=*/{padding, padding + 1},
                                           /*use_full_conv_precision=*/true);
          } else {
            dev_output = dev_input->conv2d(*dev_weight,
                                           /*stride=*/{stride, stride + 1},
                                           /*padding=*/{padding, padding + 1},
                                           /*use_full_conv_precision=*/true);
          }
          AllClose(output, *dev_output);
        });
      }
    }
  }
}

}  // namespace cpp_test
}  // namespace torch_xla
