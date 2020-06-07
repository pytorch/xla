#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "cpp_test_util.h"
#include "torch/csrc/autograd/variable.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla_test.h"

namespace torch_xla {
namespace cpp_test {
namespace {

bool CheckBidirectionalConversion(
    const at::Tensor& input, at::ScalarType dest_element_type,
    c10::optional<xla::PrimitiveType> xla_type = c10::nullopt) {
  xla::Literal literal =
      GetTensorLiteral(input, /*shape=*/nullptr, /*device=*/nullptr);
  if (xla_type) {
    literal = literal.Convert(*xla_type).ConsumeValueOrDie();
  }
  at::Tensor converted = MakeTensorFromXlaLiteral(literal, dest_element_type);
  return EqualValuesNoElementTypeCheck(converted, input);
}

}  // namespace

using TensorTest = TorchXlaTest;

TEST_F(TensorTest, TestConversions) {
  {
    at::Tensor a = at::randint(std::numeric_limits<uint8_t>::min(),
                               std::numeric_limits<uint8_t>::max(), {2, 2},
                               at::TensorOptions(at::kByte));
    EXPECT_TRUE(CheckBidirectionalConversion(a, at::ScalarType::Short));
    EXPECT_TRUE(CheckBidirectionalConversion(a, at::ScalarType::Int));
    EXPECT_TRUE(CheckBidirectionalConversion(a, at::ScalarType::Long));
  }
  {
    at::Tensor a = at::randint(std::numeric_limits<int8_t>::min(),
                               std::numeric_limits<int8_t>::max(), {2, 2},
                               at::TensorOptions(at::kChar));
    EXPECT_TRUE(CheckBidirectionalConversion(a, at::ScalarType::Short));
    EXPECT_TRUE(CheckBidirectionalConversion(a, at::ScalarType::Int));
    EXPECT_TRUE(CheckBidirectionalConversion(a, at::ScalarType::Long));
  }
  {
    at::Tensor a = at::randint(std::numeric_limits<int16_t>::min(),
                               std::numeric_limits<int16_t>::max(), {2, 2},
                               at::TensorOptions(at::kShort));
    EXPECT_TRUE(CheckBidirectionalConversion(a, at::ScalarType::Int));
    EXPECT_TRUE(CheckBidirectionalConversion(a, at::ScalarType::Long));
  }
  {
    at::Tensor a = at::randint(std::numeric_limits<int32_t>::min(),
                               std::numeric_limits<int32_t>::max(), {2, 2},
                               at::TensorOptions(at::kInt));
    EXPECT_TRUE(CheckBidirectionalConversion(a, at::ScalarType::Long));
  }
  {
    at::Tensor a = at::randint(0, 1, {2, 2}, at::TensorOptions(at::kByte));
    EXPECT_TRUE(CheckBidirectionalConversion(a, at::ScalarType::Byte,
                                             xla::PrimitiveType::PRED));
  }
  {
    at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    EXPECT_TRUE(CheckBidirectionalConversion(a, at::ScalarType::Double));
  }
}

TEST_F(TensorTest, TestAdd) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = a.add(b, 1.0);

  ForEachDevice([&](const Device& device) {
    XLATensor dev_a = XLATensor::Create(a, device);
    XLATensor dev_b = XLATensor::Create(b, device);
    XLATensor dev_c = XLATensor::add(dev_a, dev_b, 1.0);

    AllClose(c, dev_c);
  });
}

TEST_F(TensorTest, TestIntegerAdd) {
  std::vector<at::ScalarType> types(
      {at::kByte, at::kChar, at::kShort, at::kInt, at::kLong});

  ForEachDevice([&](const Device& device) {
    for (auto type : types) {
      at::Tensor a = at::randint(0, 63, {2, 2}, at::TensorOptions(type));
      at::Tensor b = at::randint(0, 63, {2, 2}, at::TensorOptions(type));
      at::Scalar one =
          at::isIntegralType(type) ? at::Scalar(int64_t(1)) : at::Scalar(1.0);
      at::Tensor c = a.add(b, one);

      XLATensor dev_a = XLATensor::Create(a, device);
      XLATensor dev_b = XLATensor::Create(b, device);
      XLATensor dev_c = XLATensor::add(dev_a, dev_b, one);

      EXPECT_TRUE(
          EqualValuesNoElementTypeCheck(c, dev_c.ToTensor(/*detached=*/false)));
    }
  });
}

TEST_F(TensorTest, TestSize) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  ForEachDevice([&](const Device& device) {
    XLATensor dev_input = XLATensor::Create(input, device);
    for (int dim = -rank; dim < rank; ++dim) {
      EXPECT_EQ(input.size(dim), dev_input.size(dim));
    }
  });
}

TEST_F(TensorTest, TestRelu) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  at::Tensor output = input.relu();
  ForEachDevice([&](const Device& device) {
    XLATensor dev_input = XLATensor::Create(input, device);
    XLATensor dev_output = XLATensor::relu(dev_input);
    AllClose(output, dev_output);
  });
}

TEST_F(TensorTest, TestRrelu) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  float lower = 0.125;
  float upper = 0.5;
  ForEachDevice([&](const Device& device) {
    for (bool training : {true, false}) {
      at::Tensor noise = at::zeros_like(input);
      at::Tensor output =
          at::rrelu_with_noise(input, noise, lower, upper, training);
      XLATensor dev_input = XLATensor::Create(input, device);
      XLATensor dev_noise = XLATensor::Create(noise, device);
      XLATensor dev_outputs = XLATensor::rrelu_with_noise(
          dev_input, dev_noise, lower, upper, training);
      AllClose(output, dev_outputs);
      AllClose(noise, dev_noise);
    }
  });
}

TEST_F(TensorTest, TestThreshold) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  float threshold = 0.4;
  float value = 20;
  at::Tensor output = at::threshold(input, threshold, value);
  ForEachDevice([&](const Device& device) {
    XLATensor dev_input = XLATensor::Create(input, device);
    XLATensor dev_output = XLATensor::threshold(dev_input, threshold, value);
    AllClose(output, dev_output);
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
  at::Tensor output = at::addmm(bias, input, weight);
  ForEachDevice([&](const Device& device) {
    XLATensor dev_input = XLATensor::Create(input, device);
    XLATensor dev_weight = XLATensor::Create(weight, device);
    XLATensor dev_bias = XLATensor::Create(bias, device);
    XLATensor dev_output = XLATensor::addmm(dev_input, dev_weight, dev_bias);
    AllClose(output, dev_output);
  });
}

TEST_F(TensorTest, TestTranspose) {
  at::Tensor input = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::transpose(input, 0, 1);
  ForEachDevice([&](const Device& device) {
    XLATensor dev_input = XLATensor::Create(input, device);
    XLATensor dev_output = XLATensor::transpose(dev_input, 0, 1);
    AllClose(output, dev_output);
  });
}

TEST_F(TensorTest, TestView) {
  at::Tensor input = at::rand({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor output = input.view({-1, 320});
  ForEachDevice([&](const Device& device) {
    XLATensor dev_input = XLATensor::Create(input, device);
    XLATensor dev_output = XLATensor::view(dev_input, {-1, 320});
    AllClose(output, dev_output);
  });
}

TEST_F(TensorTest, TestViewMod) {
  at::Tensor input = at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor one = at::tensor(1.0, at::TensorOptions(at::kFloat));
  at::Tensor output = input.view({-1, 320});
  output.add_(one, 1.0);
  input.add_(one, 1.0);
  ForEachDevice([&](const Device& device) {
    at::Tensor xinput =
        at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
    XLATensor dev_input = XLATensor::Create(xinput, device);
    XLATensor dev_one = XLATensor::Create(one, device);
    XLATensor dev_output = XLATensor::view(dev_input, {-1, 320});
    XLATensor::add_(dev_output, dev_one, 1.0);
    XLATensor::add_(dev_input, dev_one, 1.0);
    AllClose(output, dev_output);
    AllClose(input, dev_input);
  });
}

TEST_F(TensorTest, TestViewModComplex) {
  at::Tensor input = at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor one = at::tensor(1.0, at::TensorOptions(at::kFloat));
  at::Tensor output1 = input.view({-1, 320});
  output1.add_(one, 1.0);
  at::Tensor output2 = input.view({-1, 160});
  output2.add_(one, 1.0);
  ForEachDevice([&](const Device& device) {
    at::Tensor xinput =
        at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
    XLATensor dev_input = XLATensor::Create(xinput, device);
    XLATensor dev_one = XLATensor::Create(one, device);
    XLATensor dev_output1 = XLATensor::view(dev_input, {-1, 320});
    XLATensor::add_(dev_output1, dev_one, 1.0);
    XLATensor dev_output2 = XLATensor::view(dev_input, {-1, 160});
    XLATensor::add_(dev_output2, dev_one, 1.0);
    AllClose(output1, dev_output1);
    AllClose(output2, dev_output2);
  });
}

TEST_F(TensorTest, TestViewOfViewMod) {
  at::Tensor input = at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor one = at::tensor(1.0, at::TensorOptions(at::kFloat));
  at::Tensor output1 = input.view({-1, 320});
  output1.add_(one, 1.0);
  at::Tensor output2 = output1.view({-1, 160});
  output2.add_(one, 1.0);
  ForEachDevice([&](const Device& device) {
    at::Tensor xinput =
        at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
    XLATensor dev_input = XLATensor::Create(xinput, device);
    XLATensor dev_one = XLATensor::Create(one, device);
    XLATensor dev_output1 = XLATensor::view(dev_input, {-1, 320});
    XLATensor::add_(dev_output1, dev_one, 1.0);
    XLATensor dev_output2 = XLATensor::view(dev_output1, {-1, 160});
    XLATensor::add_(dev_output2, dev_one, 1.0);
    AllClose(output1, dev_output1);
    AllClose(output2, dev_output2);
  });
}

TEST_F(TensorTest, TestLogSoftmax) {
  at::Tensor input = at::rand({5, 3, 4, 2}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    XLATensor dev_input = XLATensor::Create(input, device);
    for (int dim = 0; dim < input.dim(); ++dim) {
      at::Tensor output = input.log_softmax(dim);
      XLATensor dev_output =
          XLATensor::log_softmax(dev_input, dim, c10::nullopt);
      AllClose(output, dev_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(TensorTest, TestMaxPool2D) {
  at::Tensor input = at::rand({1, 64, 112, 112}, at::TensorOptions(at::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      at::Tensor output =
          at::max_pool2d(input, /*kernel_size=*/{kernel_size, kernel_size},
                         /*stride=*/{stride, stride},
                         /*padding=*/{padding, padding}, /*dilation=*/{1, 1},
                         /*ceil_mode=*/false);
      ForEachDevice([&](const Device& device) {
        XLATensor dev_input = XLATensor::Create(input, device);
        auto dev_output = XLATensor::max_pool_nd(
            dev_input,
            /*spatial_dim_count=*/2,
            /*kernel_size=*/{kernel_size, kernel_size},
            /*stride=*/{stride, stride},
            /*padding=*/{padding, padding}, /*ceil_mode=*/false);
        AllClose(output, std::get<0>(dev_output));
      });
    }
  }
}

TEST_F(TensorTest, TestMaxPool2DNonSquare) {
  at::Tensor input = at::rand({1, 64, 112, 112}, at::TensorOptions(at::kFloat));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      at::Tensor output = at::max_pool2d(
          input, /*kernel_size=*/{kernel_size, kernel_size + 1},
          /*stride=*/{stride, stride + 1},
          /*padding=*/{padding, padding + 1}, /*dilation=*/{1, 1},
          /*ceil_mode=*/false);
      ForEachDevice([&](const Device& device) {
        XLATensor dev_input = XLATensor::Create(input, device);
        auto dev_output = XLATensor::max_pool_nd(
            dev_input,
            /*spatial_dim_count=*/2,
            /*kernel_size=*/{kernel_size, kernel_size + 1},
            /*stride=*/{stride, stride + 1},
            /*padding=*/{padding, padding + 1},
            /*ceil_mode=*/false);
        AllClose(output, std::get<0>(dev_output));
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
        at::Tensor output =
            at::avg_pool2d(input,
                           /*kernel_size=*/{kernel_size, kernel_size},
                           /*stride=*/{stride, stride},
                           /*padding=*/{padding, padding},
                           /*ceil_mode=*/false, count_include_pad);
        ForEachDevice([&](const Device& device) {
          XLATensor dev_input = XLATensor::Create(input, device);
          XLATensor dev_output =
              XLATensor::avg_pool_nd(dev_input,
                                     /*spatial_dim_count=*/2,
                                     /*kernel_size=*/{kernel_size, kernel_size},
                                     /*stride=*/{stride, stride},
                                     /*padding=*/{padding, padding},
                                     /*ceil_mode=*/false, count_include_pad);
          AllClose(output, dev_output);
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
        at::Tensor output = at::avg_pool2d(
            input,
            /*kernel_size=*/{kernel_size, kernel_size + 1},
            /*stride=*/{stride, stride + 1},
            /*padding=*/{padding, padding + 1}, /*ceil_mode=*/false,
            /*count_include_pad=*/count_include_pad);
        ForEachDevice([&](const Device& device) {
          XLATensor dev_input = XLATensor::Create(input, device);
          XLATensor dev_output = XLATensor::avg_pool_nd(
              dev_input,
              /*spatial_dim_count=*/2,
              /*kernel_size=*/{kernel_size, kernel_size + 1},
              /*stride=*/{stride, stride + 1},
              /*padding=*/{padding, padding + 1},
              /*ceil_mode=*/false,
              /*count_include_pad=*/count_include_pad);
          AllClose(output, dev_output);
        });
      }
    }
  }
}

TEST_F(TensorTest, TestBatchNorm1D) {
  int num_features = 3;
  at::Tensor input =
      at::rand({2, num_features, 4}, at::TensorOptions(at::kFloat));
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
    for (bool undef_weight_bias : {true, false}) {
      auto output = at::native_batch_norm(
          /*input=*/input, /*weight=*/undef_weight_bias ? undef : weight,
          /*bias=*/undef_weight_bias ? undef : bias,
          /*running_mean=*/running_mean, /*running_var=*/running_var,
          /*training=*/training, /*momentum=*/momentum, /*eps=*/eps);
      ForEachDevice([&](const Device& device) {
        XLATensor xla_input = XLATensor::Create(input, device);
        XLATensor xla_weight =
            undef_weight_bias ? XLATensor() : XLATensor::Create(weight, device);
        XLATensor xla_bias =
            undef_weight_bias ? XLATensor() : XLATensor::Create(bias, device);
        XLATensor xla_running_mean = XLATensor::Create(running_mean, device);
        XLATensor xla_running_var = XLATensor::Create(running_var, device);
        auto xla_output = XLATensor::native_batch_norm(
            /*input=*/xla_input, /*weight=*/xla_weight, /*bias=*/xla_bias,
            /*running_mean=*/xla_running_mean, /*running_var=*/xla_running_var,
            /*training=*/training, /*momentum=*/momentum, /*eps=*/eps);
        AllClose(std::get<0>(output), std::get<0>(xla_output));
        // native_batch_norm return undefined for save_mean & save_invstd when
        // training=false.
        EXPECT_EQ(std::get<1>(output).defined(),
                  !std::get<1>(xla_output).is_null());
        EXPECT_EQ(std::get<2>(output).defined(),
                  !std::get<2>(xla_output).is_null());
        if (training) {
          AllClose(std::get<1>(output), std::get<1>(xla_output));
          AllClose(std::get<2>(output), std::get<2>(xla_output));
        }
      });
    }
  }
}

TEST_F(TensorTest, TestConv2D) {
  int in_channels = 9;
  int out_channels = 3;
  int kernel_size = 5;
  at::Tensor input =
      at::rand({4, in_channels, 32, 32}, at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({out_channels}, at::TensorOptions(at::kFloat));
  at::Tensor no_bias;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        for (int dilation = 1; dilation <= 2; ++dilation) {
          for (int groups : {1, 3}) {
            for (bool transposed : {true, false}) {
              for (int output_padding = 0;
                   output_padding < std::min(stride, dilation);
                   ++output_padding) {
                at::Tensor weight =
                    transposed ? at::rand({in_channels, out_channels / groups,
                                           kernel_size, kernel_size})
                               : at::rand({out_channels, in_channels / groups,
                                           kernel_size, kernel_size},
                                          at::TensorOptions(at::kFloat));

                at::Tensor output = at::native::_convolution(
                    input, weight, with_bias ? bias : no_bias,
                    /*stride=*/{stride, stride},
                    /*padding=*/{padding, padding},
                    /*dilation=*/{dilation, dilation},
                    /*transposed=*/transposed,
                    /*output_padding=*/{output_padding, output_padding},
                    /*groups=*/groups, false, false, false);
                ForEachDevice([&](const Device& device) {
                  XLATensor dev_input = XLATensor::Create(input, device);
                  XLATensor dev_weight = XLATensor::Create(weight, device);
                  XLATensor dev_output;
                  if (with_bias) {
                    XLATensor dev_bias = XLATensor::Create(bias, device);
                    dev_output = XLATensor::convolution_overrideable(
                        dev_input, dev_weight, dev_bias,
                        /*stride=*/{stride, stride},
                        /*padding=*/{padding, padding},
                        /*dilation=*/{dilation, dilation},
                        /*transposed=*/transposed,
                        /*output_padding=*/{output_padding, output_padding},
                        /*groups=*/groups);
                  } else {
                    dev_output = XLATensor::convolution_overrideable(
                        dev_input, dev_weight,
                        /*stride=*/{stride, stride},
                        /*padding=*/{padding, padding},
                        /*dilation=*/{dilation, dilation},
                        /*transposed=*/transposed,
                        /*output_padding=*/{output_padding, output_padding},
                        /*groups=*/groups);
                  }
                  AllClose(output, dev_output, /*rtol=*/5e-3, /*atol=*/1e-3);
                });
              };
            }
          }
        }
      }
    }
  }
}

TEST_F(TensorTest, TestConv2DNonSquare) {
  int in_channels = 3;
  int out_channels = 6;
  int kernel_size = 5;
  at::Tensor input =
      at::rand({4, in_channels, 26, 26}, at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({out_channels}, at::TensorOptions(at::kFloat));
  at::Tensor no_bias;

  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 0; ++padding) {
      for (bool with_bias : {true, false}) {
        for (int dilation = 1; dilation <= 2; ++dilation) {
          for (int groups : {1, 3}) {
            for (bool transposed : {true, false}) {
              for (int output_padding = 0;
                   output_padding < std::min(stride, dilation);
                   ++output_padding) {
                at::Tensor weight =
                    transposed ? at::rand({in_channels, out_channels / groups,
                                           kernel_size, kernel_size})
                               : at::rand({out_channels, in_channels / groups,
                                           kernel_size, kernel_size},
                                          at::TensorOptions(at::kFloat));

                at::Tensor output = at::native::_convolution(
                    input, weight, with_bias ? bias : no_bias,
                    /*stride=*/{stride, stride + 1},
                    /*padding=*/{padding, padding + 1},
                    /*dilation=*/{dilation, dilation + 1},
                    /*transposed=*/transposed,
                    /*output_padding=*/{output_padding, output_padding + 1},
                    /*groups=*/groups, false, false, false);

                ForEachDevice([&](const Device& device) {
                  XLATensor dev_input = XLATensor::Create(input, device);
                  XLATensor dev_weight = XLATensor::Create(weight, device);
                  XLATensor dev_output;
                  if (with_bias) {
                    XLATensor dev_bias = XLATensor::Create(bias, device);
                    dev_output = XLATensor::convolution_overrideable(
                        dev_input, dev_weight, dev_bias,
                        /*stride=*/{stride, stride + 1},
                        /*padding=*/{padding, padding + 1},
                        /*dilation=*/{dilation, dilation + 1},
                        /*transposed=*/transposed,
                        /*output_padding=*/{output_padding, output_padding + 1},
                        /*groups=*/groups);

                  } else {
                    dev_output = XLATensor::convolution_overrideable(
                        dev_input, dev_weight,
                        /*stride=*/{stride, stride + 1},
                        /*padding=*/{padding, padding + 1},
                        /*dilation=*/{dilation, dilation + 1},
                        /*transposed=*/transposed,
                        /*output_padding=*/{output_padding, output_padding + 1},
                        /*groups=*/groups);
                  }
                  AllClose(output, dev_output, /*rtol=*/5e-3, /*atol=*/1e-3);
                });
              }
            }
          }
        }
      }
    }
  }
}

TEST_F(TensorTest, TestConv3D) {
  int in_channels = 9;
  int out_channels = 3;
  int kernel_size = 5;
  at::Tensor input =
      at::rand({4, in_channels, 28, 28, 28}, at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({out_channels}, at::TensorOptions(at::kFloat));
  at::Tensor no_bias;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        for (int dilation = 1; dilation <= 1; ++dilation) {
          for (int groups : {1, 3}) {
            for (bool transposed : {true, false}) {
              for (int output_padding = 0;
                   output_padding < std::min(stride, dilation);
                   ++output_padding) {
                at::Tensor weight =
                    transposed
                        ? at::rand({in_channels, out_channels / groups,
                                    kernel_size, kernel_size, kernel_size})
                        : at::rand({out_channels, in_channels / groups,
                                    kernel_size, kernel_size, kernel_size},
                                   at::TensorOptions(at::kFloat));

                at::Tensor output = at::native::_convolution(
                    input, weight, with_bias ? bias : no_bias,
                    /*stride=*/{stride, stride, stride},
                    /*padding=*/{padding, padding, padding},
                    /*dilation=*/{dilation, dilation, dilation},
                    /*transposed=*/transposed,
                    /*output_padding=*/
                    {output_padding, output_padding, output_padding},
                    /*groups=*/groups, false, false, false);
                ForEachDevice([&](const Device& device) {
                  XLATensor dev_input = XLATensor::Create(input, device);
                  XLATensor dev_weight = XLATensor::Create(weight, device);
                  XLATensor dev_output;
                  if (with_bias) {
                    XLATensor dev_bias = XLATensor::Create(bias, device);
                    dev_output = XLATensor::convolution_overrideable(
                        dev_input, dev_weight, dev_bias,
                        /*stride=*/{stride, stride, stride},
                        /*padding=*/{padding, padding, padding},
                        /*dilation=*/{dilation, dilation, dilation},
                        /*transposed=*/transposed,
                        /*output_padding=*/
                        {output_padding, output_padding, output_padding},
                        /*groups=*/groups);
                  } else {
                    dev_output = XLATensor::convolution_overrideable(
                        dev_input, dev_weight,
                        /*stride=*/{stride, stride, stride},
                        /*padding=*/{padding, padding, padding},
                        /*dilation=*/{dilation, dilation, dilation},
                        /*transposed=*/transposed,
                        /*output_padding=*/
                        {output_padding, output_padding, output_padding},
                        /*groups=*/groups);
                  }
                  AllClose(output, dev_output, /*rtol=*/5e-3, /*atol=*/1e-3);
                });
              };
            }
          }
        }
      }
    }
  }
}

TEST_F(TensorTest, TestConv3DNonSquare) {
  int in_channels = 9;
  int out_channels = 3;
  int kernel_size = 5;
  at::Tensor input =
      at::rand({4, in_channels, 28, 28, 28}, at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({out_channels}, at::TensorOptions(at::kFloat));
  at::Tensor no_bias;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        for (int dilation = 1; dilation <= 1; ++dilation) {
          for (int groups : {1, 3}) {
            for (bool transposed : {true, false}) {
              for (int output_padding = 0;
                   output_padding < std::min(stride, dilation);
                   ++output_padding) {
                at::Tensor weight =
                    transposed
                        ? at::rand({in_channels, out_channels / groups,
                                    kernel_size, kernel_size, kernel_size})
                        : at::rand({out_channels, in_channels / groups,
                                    kernel_size, kernel_size, kernel_size},
                                   at::TensorOptions(at::kFloat));

                at::Tensor output = at::native::_convolution(
                    input, weight, with_bias ? bias : no_bias,
                    /*stride=*/{stride, stride + 1, stride + 1},
                    /*padding=*/{padding, padding + 1, padding + 1},
                    /*dilation=*/{dilation, dilation + 1, dilation + 1},
                    /*transposed=*/transposed,
                    /*output_padding=*/
                    {output_padding, output_padding + 1, output_padding},
                    /*groups=*/groups, false, false, false);
                ForEachDevice([&](const Device& device) {
                  XLATensor dev_input = XLATensor::Create(input, device);
                  XLATensor dev_weight = XLATensor::Create(weight, device);
                  XLATensor dev_output;
                  if (with_bias) {
                    XLATensor dev_bias = XLATensor::Create(bias, device);
                    dev_output = XLATensor::convolution_overrideable(
                        dev_input, dev_weight, dev_bias,
                        /*stride=*/{stride, stride + 1, stride + 1},
                        /*padding=*/{padding, padding + 1, padding + 1},
                        /*dilation=*/{dilation, dilation + 1, dilation + 1},
                        /*transposed=*/transposed,
                        /*output_padding=*/
                        {output_padding, output_padding + 1, output_padding},
                        /*groups=*/groups);
                  } else {
                    dev_output = XLATensor::convolution_overrideable(
                        dev_input, dev_weight,
                        /*stride=*/{stride, stride + 1, stride + 1},
                        /*padding=*/{padding, padding + 1, padding + 1},
                        /*dilation=*/{dilation, dilation + 1, dilation + 1},
                        /*transposed=*/transposed,
                        /*output_padding=*/
                        {output_padding, output_padding + 1, output_padding},
                        /*groups=*/groups);
                  }
                  AllClose(output, dev_output, /*rtol=*/5e-3, /*atol=*/1e-3);
                });
              };
            }
          }
        }
      }
    }
  }
}

}  // namespace cpp_test
}  // namespace torch_xla
