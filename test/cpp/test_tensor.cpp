#include <gtest/gtest.h>

#include <vector>

#include <ATen/ATen.h>
#include "cpp_test_util.h"
#include "tensor.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_xla {
namespace cpp_test {

TEST(TensorTest, TestAdd) {
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

TEST(TensorTest, TestIntegerAdd) {
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

TEST(TensorTest, TestConv2D) {
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

TEST(TensorTest, TestConv2DNonSquare) {
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
