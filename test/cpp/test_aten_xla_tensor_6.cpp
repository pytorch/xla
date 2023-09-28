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

TEST_F(AtenXlaTensorTest, TestTransposedConv3DBackward) {
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
                return torch::conv_transpose3d(
                    inputs[0], inputs[1], inputs[2],
                    /*stride=*/{stride, stride + 1, stride},
                    /*padding=*/{padding, padding + 1, stride},
                    /*output_padding=*/output_padding,
                    /*groups=*/groups,
                    /*dilation=*/{dilation, dilation + 1, dilation});
              };
              ForEachDevice([&](const torch::Device& device) {
                torch::Tensor input = torch::rand(
                    {4, out_channels, 14, 14, 14},
                    torch::TensorOptions(torch::kDouble).requires_grad(true));
                torch::Tensor weight = torch::rand(
                    {out_channels, in_channels / groups, kernel_size,
                     kernel_size, kernel_size},
                    torch::TensorOptions(torch::kDouble).requires_grad(true));
                torch::Tensor bias =
                    with_bias ? torch::rand({in_channels},
                                            torch::TensorOptions(torch::kDouble)
                                                .requires_grad(true))
                              : torch::Tensor();
                TestBackward({input, weight, bias}, device, testfn);
              });
            }
          };
        }
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
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          return torch::max_pool2d(
              inputs[0], /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const torch::Device& device) {
          TestBackward(
              {torch::rand(
                  {1, 64, 112, 112},
                  torch::TensorOptions(torch::kFloat).requires_grad(true))},
              device, testfn);
        });

        ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
        ExpectCounterChanged("xla::max_pool2d", cpp_test::GetIgnoredCounters());
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
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          return torch::max_pool3d(
              inputs[0],
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding}, /*dilation=*/{1, 1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const torch::Device& device) {
          TestBackward(
              {torch::rand(
                  {1, 64, 16, 16, 16},
                  torch::TensorOptions(torch::kFloat).requires_grad(true))},
              device, testfn);
        });

        ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
        ExpectCounterChanged("xla::max_pool3d", cpp_test::GetIgnoredCounters());
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool2DNoBatchBackward) {
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          return torch::max_pool2d(
              inputs[0], /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const torch::Device& device) {
          TestBackward(
              {torch::rand(
                  {64, 112, 112},
                  torch::TensorOptions(torch::kFloat).requires_grad(true))},
              device, testfn);
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool3DNoBatchBackward) {
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          return torch::max_pool3d(
              inputs[0],
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding}, /*dilation=*/{1, 1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const torch::Device& device) {
          TestBackward(
              {torch::rand(
                  {64, 16, 16, 16},
                  torch::TensorOptions(torch::kFloat).requires_grad(true))},
              device, testfn);
        });

        ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
        ExpectCounterChanged("xla::max_pool3d", cpp_test::GetIgnoredCounters());
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxUnpool2DBackward) {
  int kernel_size = 2;
  torch::Tensor input =
      torch::rand({2, 2, 8, 8}, torch::TensorOptions(torch::kFloat));
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output;
          torch::Tensor indices;
          std::tie(output, indices) = torch::max_pool2d_with_indices(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          std::vector<int64_t> output_size({input.size(2), input.size(3)});
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::max_unpool2d(inputs[0], inputs[1], output_size);
          };

          ForEachDevice([&](const torch::Device& device) {
            TestBackward({output.requires_grad_(true), indices}, device,
                         testfn);
          });
        }
      }
    }
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestMaxUnpool3DBackward) {
  int kernel_size = 2;
  torch::Tensor input =
      torch::rand({2, 2, 8, 8, 8}, torch::TensorOptions(torch::kFloat));
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
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
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::max_unpool3d(inputs[0], inputs[1], output_size,
                                       /*stride=*/{stride, stride, stride},
                                       /*padding=*/{padding, padding, padding});
          };

          ForEachDevice([&](const torch::Device& device) {
            TestBackward({output.requires_grad_(true), indices}, device,
                         testfn);
          });
        }
      }
    }
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestTanhBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::tanh(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 2},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestSigmoidBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::sigmoid(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 2},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestLogSigmoidBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::log_sigmoid(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 2},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn, /*rtol=*/1e-3, /*atol=*/1e-5);
  });

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::log_sigmoid_forward",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLogSoftmaxBackward) {
  for (int dim = -4; dim < 4; ++dim) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::log_softmax(inputs[0], dim);
    };

    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {5, 3, 4, 2},
              torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
    });

    ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::_log_softmax", cpp_test::GetIgnoredCounters());
  }
}

TEST_F(AtenXlaTensorTest, TestSoftmaxBackward) {
  for (int dim = -4; dim < 4; ++dim) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::softmax(inputs[0], dim);
    };

    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {5, 3, 4, 2},
              torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestSoftplusBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::softplus(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 1, 4, 6},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn, /*rtol=*/1e-4);
  });
}

TEST_F(AtenXlaTensorTest, TestReluBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::relu(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 1, 4, 6},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestRreluBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::rrelu(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 1, 4, 6},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestHardshrinkBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::hardshrink(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::randn({100},
                      torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestHardshrinkBackwardWithMixedDataType) {
  if (UsingTpu()) {
    GTEST_SKIP();
  }
  torch::Tensor lambdaTensor =
      torch::scalar_tensor(0., torch::TensorOptions(torch::kFloat32));
  torch::Scalar lambda = lambdaTensor.item();

  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::hardshrink(inputs[0], lambda);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::randn(
            {100}, torch::TensorOptions(torch::kFloat64).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestSoftshrinkBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::softshrink(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::randn({100},
                      torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestSoftshrinkBackwardWithMixedDataType) {
  if (UsingTpu()) {
    GTEST_SKIP();
  }
  torch::Tensor lambdaTensor =
      torch::scalar_tensor(0., torch::TensorOptions(torch::kFloat32));
  torch::Scalar lambda = lambdaTensor.item();

  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::softshrink(inputs[0], lambda);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::randn(
            {100}, torch::TensorOptions(torch::kFloat64).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestHardtanhBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::hardtanh(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::randn({100},
                      torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestEluBackward) {
  torch::Scalar alpha = 0.5;
  torch::Scalar scale = 2.5;
  torch::Scalar input_scale = 1.5;
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::elu(inputs[0], alpha, scale, input_scale);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 1, 4, 6},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestGeluBackward) {
  for (const auto& approximate : {"none", "tanh"}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::gelu(inputs[0], approximate);
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {2, 3}, torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device, testfn);
    });
    ExpectCounterChanged("xla::gelu_backward", cpp_test::GetIgnoredCounters());
  }
}

TEST_F(AtenXlaTensorTest, TestLeakyReluBackward) {
  double negative_slope = 0.01;
  auto testfn = [=](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::leaky_relu(inputs[0], negative_slope);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 1, 4, 6},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestTransposeBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::t(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 3},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestAddMatMulBackward) {
  int in_channels = 32;
  int out_channels = 320;
  int labels = 50;
  // Test beta != 1. through the CPU interop.
  for (double beta : {1., 2.}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::addmm(inputs[0], inputs[1], inputs[2], /*beta=*/beta);
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand({labels},
                       torch::TensorOptions(torch::kFloat).requires_grad(true)),
           torch::rand({in_channels, out_channels},
                       torch::TensorOptions(torch::kFloat).requires_grad(true)),
           torch::rand(
               {out_channels, labels},
               torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device, testfn);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestBinaryCrossEntropyBackward) {
  if (UsingTpu()) {
    GTEST_SKIP();
  }
  int batch = 6;
  int classes = 2;
  for (auto dtype : {torch::kFloat, torch::kDouble}) {
    for (bool def_weight : {false, true}) {
      torch::Tensor input = torch::rand(
          {batch, classes}, torch::TensorOptions(dtype).requires_grad(true));
      torch::Tensor target =
          torch::rand({batch, classes}, torch::TensorOptions(dtype));
      torch::Tensor weight;
      if (def_weight) {
        weight = torch::rand({batch, classes}, torch::TensorOptions(dtype));
      }
      for (torch::Reduction::Reduction reduction :
           {torch::Reduction::Mean, torch::Reduction::Sum,
            torch::Reduction::None}) {
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          return torch::binary_cross_entropy(
              /*self=*/inputs[0], /*target=*/inputs[1],
              /*weight=*/inputs[2],
              /*reduction=*/reduction);
        };
        ForEachDevice([&](const torch::Device& device) {
          TestBackward({input, target, weight}, device, testfn, /*rtol=*/1e-4,
                       /*atol=*/1e-7);
        });
      }
    }
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::binary_cross_entropy",
                       cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::binary_cross_entropy_backward",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestNllLossBackward) {
  int batch = 6;
  int classes = 2;
  for (auto dtype : {torch::kFloat, torch::kDouble}) {
    for (int ignore_index : {-1, 0, 1, 5}) {
      for (bool def_weight : {false, true}) {
        torch::Tensor input = torch::rand(
            {batch, classes}, torch::TensorOptions(dtype).requires_grad(true));
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
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::nll_loss(
                /*self=*/inputs[0], /*target=*/inputs[1],
                /*weight=*/inputs[2],
                /*reduction=*/reduction, /*ignore_index=*/ignore_index);
          };
          ForEachDevice([&](const torch::Device& device) {
            TestBackward({input, target, weight}, device, testfn, /*rtol=*/1e-5,
                         /*atol=*/1e-8);
          });
        }
      }
    }
  }

  ExpectCounterNotChanged("aten::(?!_local_scalar_dense).*",
                          cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::nll_loss_forward", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::nll_loss_backward",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestNllLoss2dBackward) {
  int batch = 6;
  int classes = 2;
  int height = 3;
  int width = 3;
  for (auto dtype : {torch::kFloat, torch::kDouble}) {
    for (int ignore_index : {-1, 0, 1, 5}) {
      for (bool def_weight : {false, true}) {
        torch::Tensor input =
            torch::rand({batch, classes, height, width},
                        torch::TensorOptions(dtype).requires_grad(true));
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
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::nll_loss2d(
                /*self=*/inputs[0], /*target=*/inputs[1],
                /*weight=*/inputs[2],
                /*reduction=*/reduction, /*ignore_index=*/ignore_index);
          };
          ForEachDevice([&](const torch::Device& device) {
            TestBackward({input, target, weight}, device, testfn, /*rtol=*/1e-5,
                         /*atol=*/1e-8);
          });
        }
      }
    }
  }

  ExpectCounterNotChanged("aten::(?!_local_scalar_dense).*",
                          cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::nll_loss2d_forward",
                       cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::nll_loss2d_backward",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestSmoothL1LossBackward) {
  torch::Tensor input = torch::randn(
      {2, 4}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  torch::Tensor target =
      torch::randn({2, 4}, torch::TensorOptions(torch::kFloat));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None, torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    for (double beta : {0.25, 1.}) {
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::smooth_l1_loss(/*input=*/inputs[0], /*target=*/inputs[1],
                                     /*reduction=*/reduction, /*beta=*/beta);
      };
      ForEachDevice([&](const torch::Device& device) {
        TestBackward({input, target}, device, testfn, /*rtol=*/1e-5,
                     /*atol=*/1e-8);
      });
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::smooth_l1_loss_backward",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestViewBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return inputs[0].view({-1, 320});
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({32, 20, 4, 4},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestBatchNorm2DBackward) {
  double momentum = 0.1;
  double eps = 0.5;
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::batch_norm(
        /*input=*/inputs[0], /*weight=*/inputs[1], /*bias=*/inputs[2],
        /*running_mean=*/inputs[3], /*running_var=*/inputs[4],
        /*training=*/true, /*momentum=*/momentum, /*eps=*/eps,
        /*cudnn_enabled=*/false);
  };
  int num_features = 3;
  torch::Tensor undef;
  for (bool undef_weight_bias : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor input =
          torch::rand({2, num_features, 4, 4},
                      torch::TensorOptions(torch::kFloat).requires_grad(true));
      torch::Tensor weight =
          undef_weight_bias
              ? undef
              : torch::rand(
                    {num_features},
                    torch::TensorOptions(torch::kFloat).requires_grad(true));
      torch::Tensor bias =
          undef_weight_bias
              ? undef
              : torch::rand(
                    {num_features},
                    torch::TensorOptions(torch::kFloat).requires_grad(true));
      torch::Tensor running_mean =
          torch::zeros({num_features}, torch::TensorOptions(torch::kFloat));
      torch::Tensor running_var =
          torch::ones({num_features}, torch::TensorOptions(torch::kFloat));
      TestBackward({input, weight, bias, running_mean, running_var}, device,
                   testfn,
                   /*rtol=*/1e-3, /*atol=*/1e-4);
    });

    ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::native_batch_norm",
                         cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::native_batch_norm_backward",
                         cpp_test::GetIgnoredCounters());
  }
}

TEST_F(AtenXlaTensorTest, TestBatchNorm3DBackward) {
  double momentum = 0.1;
  double eps = 0.5;
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::batch_norm(
        /*input=*/inputs[0], /*weight=*/inputs[1], /*bias=*/inputs[2],
        /*running_mean=*/inputs[3], /*running_var=*/inputs[4],
        /*training=*/true, /*momentum=*/momentum, /*eps=*/eps,
        /*cudnn_enabled=*/false);
  };
  int num_features = 3;
  torch::Tensor undef;
  for (bool undef_weight_bias : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor input =
          torch::rand({2, num_features, 4, 4, 2},
                      torch::TensorOptions(torch::kFloat).requires_grad(true));
      torch::Tensor weight =
          undef_weight_bias
              ? undef
              : torch::rand(
                    {num_features},
                    torch::TensorOptions(torch::kFloat).requires_grad(true));
      torch::Tensor bias =
          undef_weight_bias
              ? undef
              : torch::rand(
                    {num_features},
                    torch::TensorOptions(torch::kFloat).requires_grad(true));
      torch::Tensor running_mean =
          torch::zeros({num_features}, torch::TensorOptions(torch::kFloat));
      torch::Tensor running_var =
          torch::ones({num_features}, torch::TensorOptions(torch::kFloat));
      TestBackward({input, weight, bias, running_mean, running_var}, device,
                   testfn,
                   /*rtol=*/1e-3, /*atol=*/1e-3);
    });

    ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::native_batch_norm",
                         cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::native_batch_norm_backward",
                         cpp_test::GetIgnoredCounters());
  }
}

TEST_F(AtenXlaTensorTest, TestBCEWithLogitsBackward) {
  int batch = 10;
  int classes = 5;
  torch::Tensor undef;
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None, torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::binary_cross_entropy_with_logits(
          /*input=*/inputs[0], /*target=*/inputs[1], /*weight=*/inputs[2],
          /*pos_weight=*/inputs[3],
          /*reduction=*/reduction);
    };
    for (bool undef_weight : {false, true}) {
      for (bool undef_pos_weight : {false, true}) {
        torch::Tensor input = torch::rand(
            {batch, classes},
            torch::TensorOptions(torch::kFloat).requires_grad(true));
        torch::Tensor target = torch::rand(
            {batch, classes},
            torch::TensorOptions(torch::kFloat).requires_grad(true));
        torch::Tensor weight =
            undef_weight
                ? undef
                : torch::rand({classes}, torch::TensorOptions(torch::kFloat));
        torch::Tensor pos_weight =
            undef_pos_weight
                ? undef
                : torch::rand({classes}, torch::TensorOptions(torch::kFloat));
        ForEachDevice([&](const torch::Device& device) {
          TestBackward({input, target, weight, pos_weight}, device, testfn,
                       /*rtol=*/1e-3, /*atol=*/1e-5);
        });
        ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
        // binary_cross_entropy_with_logits_backward is composed of
        // sub/mul_/add_/exp_/add_/log_/... ops in upstream pytorch.
        ExpectCounterChanged("xla::add", cpp_test::GetIgnoredCounters());
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestKlDivBackward) {
  torch::Tensor input = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  torch::Tensor target = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::kl_div(/*self=*/inputs[0], /*target=*/inputs[1], reduction);
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward({input, target}, device, testfn, /*rtol=*/1e-4,
                   /*atol=*/1e-5);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEmbeddingBackward) {
  int num_weights = 32;
  for (int padding_idx = -1; padding_idx < num_weights; ++padding_idx) {
    for (bool scale_grad_by_freq : {false, true}) {
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::embedding(inputs[0], inputs[1],
                                /*padding_idx=*/padding_idx,
                                /*scale_grad_by_freq=*/scale_grad_by_freq,
                                /*sparse=*/false);
      };
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor weight = torch::rand(
            {num_weights, 7},
            torch::TensorOptions(torch::kFloat).requires_grad(true));
        torch::Tensor indices = torch::randint(
            num_weights, {3, 9, 4}, torch::TensorOptions(torch::kLong));
        TestBackward({weight, indices}, device, testfn, /*rtol=*/1e-5,
                     /*atol=*/1e-8);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAmpUpdateScale) {
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(bridge::GetDefaultDevice()->type());
  if (hw_type != XlaDeviceType::GPU && hw_type != XlaDeviceType::CPU) {
    return;
  }
  torch::Tensor growth_tracker =
      torch::scalar_tensor(0, torch::TensorOptions(torch::kInt32));
  torch::Tensor current_scale =
      torch::scalar_tensor(4, torch::TensorOptions(torch::kFloat));
  torch::Tensor found_inf =
      torch::scalar_tensor(1, torch::TensorOptions(torch::kFloat));
  torch::Tensor not_found_inf =
      torch::scalar_tensor(0, torch::TensorOptions(torch::kFloat));
  float scale_growth_factor = 2.0;
  float scale_backoff_factor = 0.5;
  int growth_interval = 3;

  torch::Tensor growth_tracker_result0 =
      torch::scalar_tensor(1, torch::TensorOptions(torch::kInt32));
  torch::Tensor current_scale_result0 =
      torch::scalar_tensor(4, torch::TensorOptions(torch::kFloat));
  torch::Tensor growth_tracker_result1 =
      torch::scalar_tensor(2, torch::TensorOptions(torch::kInt32));
  torch::Tensor current_scale_result1 =
      torch::scalar_tensor(4, torch::TensorOptions(torch::kFloat));
  torch::Tensor growth_tracker_result2 =
      torch::scalar_tensor(0, torch::TensorOptions(torch::kInt32));
  torch::Tensor current_scale_result2 =
      torch::scalar_tensor(8, torch::TensorOptions(torch::kFloat));
  torch::Tensor growth_tracker_result3 =
      torch::scalar_tensor(0, torch::TensorOptions(torch::kInt32));
  torch::Tensor current_scale_result3 =
      torch::scalar_tensor(4, torch::TensorOptions(torch::kFloat));

  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_growth_tracker = CopyToDevice(growth_tracker, device);
    torch::Tensor xla_current_scale = CopyToDevice(current_scale, device);
    torch::Tensor xla_found_inf = CopyToDevice(found_inf, device);
    torch::Tensor xla_not_found_inf = CopyToDevice(not_found_inf, device);

    torch::_amp_update_scale_(xla_current_scale, xla_growth_tracker,
                              xla_not_found_inf, scale_growth_factor,
                              scale_backoff_factor, growth_interval);
    AllClose(current_scale_result0, xla_current_scale, /*rtol=*/1e-2,
             /*atol=*/1e-4);
    AllEqual(growth_tracker_result0, xla_growth_tracker);

    torch::_amp_update_scale_(xla_current_scale, xla_growth_tracker,
                              xla_not_found_inf, scale_growth_factor,
                              scale_backoff_factor, growth_interval);
    AllClose(current_scale_result1, xla_current_scale, /*rtol=*/1e-2,
             /*atol=*/1e-4);
    AllEqual(growth_tracker_result1, xla_growth_tracker);

    // torch::_amp_update_scale_ returns the reference of current_scale
    xla_current_scale = torch::_amp_update_scale_(
        xla_current_scale, xla_growth_tracker, xla_not_found_inf,
        scale_growth_factor, scale_backoff_factor, growth_interval);
    AllClose(current_scale_result2, xla_current_scale, /*rtol=*/1e-2,
             /*atol=*/1e-4);
    AllEqual(growth_tracker_result2, xla_growth_tracker);

    xla_current_scale = torch::_amp_update_scale_(
        xla_current_scale, xla_growth_tracker, xla_found_inf,
        scale_growth_factor, scale_backoff_factor, growth_interval);
    AllClose(current_scale_result3, xla_current_scale, /*rtol=*/1e-2,
             /*atol=*/1e-4);
    AllEqual(growth_tracker_result3, xla_growth_tracker);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_amp_update_scale_",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestEarlySyncLiveTensors) {
  torch::Tensor scalar_tensor =
      torch::scalar_tensor(1., torch::TensorOptions(torch::kFloat));
  torch::Scalar scalar1 = scalar_tensor.item();
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_scalar_tensor = CopyToDevice(scalar_tensor, device);
    torch::Scalar scalar2 = xla_scalar_tensor.item();
    ASSERT_EQ(scalar1.to<float>(), scalar2.to<float>());
  });
  if (DebugUtil::ExperimentEnabled("early_sync")) {
    ExpectCounterChanged("EarlySyncLiveTensorsCount",
                         cpp_test::GetIgnoredCounters());
  } else {
    ExpectCounterNotChanged("EarlySyncLiveTensorsCount",
                            cpp_test::GetIgnoredCounters());
  }
  ExpectCounterChanged("aten::_local_scalar_dense",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLerp) {
  torch::Tensor start =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor end = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor res = torch::lerp(start, end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_start = CopyToDevice(start, device);
    torch::Tensor xla_end = CopyToDevice(end, device);
    torch::Tensor xla_weight = CopyToDevice(weight, device);
    torch::Tensor xla_res = torch::lerp(xla_start, xla_end, xla_weight);
    AllClose(res, xla_res);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::lerp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLerpScalar) {
  torch::Tensor start =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor end = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Scalar weight = torch::Scalar(3.0);
  torch::Tensor res = torch::lerp(start, end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_start = CopyToDevice(start, device);
    torch::Tensor xla_end = CopyToDevice(end, device);
    torch::Tensor xla_res = torch::lerp(xla_start, xla_end, weight);
    AllClose(res, xla_res);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::lerp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLerpInplace) {
  torch::Tensor input =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor end = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor input_copy = input.clone();
  input.lerp_(end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input_copy, device);
    torch::Tensor xla_end = CopyToDevice(end, device);
    torch::Tensor xla_weight = CopyToDevice(weight, device);
    xla_input.lerp_(xla_end, xla_weight);
    AllClose(xla_input, input);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::lerp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLerpScalarInplace) {
  torch::Tensor input =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor end = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Scalar weight = torch::Scalar(3.0);
  torch::Tensor input_copy = input.clone();
  input.lerp_(end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input_copy, device);
    torch::Tensor xla_end = CopyToDevice(end, device);
    xla_input.lerp_(xla_end, weight);
    AllClose(xla_input, input);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::lerp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLerpOut) {
  torch::Tensor start =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor end = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor res = torch::empty({3, 4}, torch::TensorOptions(torch::kFloat));
  ;
  torch::lerp_out(res, start, end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_start = CopyToDevice(start, device);
    torch::Tensor xla_end = CopyToDevice(end, device);
    torch::Tensor xla_weight = CopyToDevice(weight, device);
    torch::Tensor xla_res = torch::empty({3, 4}, xla_start.options());
    torch::lerp_out(xla_res, xla_start, xla_end, xla_weight);
    AllClose(res, xla_res);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::lerp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLerpScalarOut) {
  torch::Tensor start =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor end = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Scalar weight = torch::Scalar(3.0);
  torch::Tensor res = torch::empty({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::lerp_out(res, start, end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_start = CopyToDevice(start, device);
    torch::Tensor xla_end = CopyToDevice(end, device);
    torch::Tensor xla_res = torch::empty({3, 4}, xla_start.options());
    torch::lerp_out(xla_res, xla_start, xla_end, weight);
    AllClose(res, xla_res);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::lerp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLinspaceStartEndMatch) {
  torch::Scalar start = 0;
  torch::Scalar end = 10;
  int64_t steps = 100;
  torch::Tensor res = torch::linspace(start, end, steps);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_res = torch::linspace(
        start, end, steps, torch::TensorOptions().device(device));
    AllClose(res, xla_res);
    AllEqual(torch::scalar_tensor(start), xla_res[0]);
    AllEqual(torch::scalar_tensor(end), xla_res[steps - 1]);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::linspace", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestLinspaceDtypes) {
  torch::Scalar start = 1;
  torch::Scalar end = 100;
  int64_t steps = 5;
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kDouble, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor res = torch::linspace(
        start, end, steps, torch::TensorOptions().dtype(scalar_type));
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_res = torch::linspace(
          start, end, steps,
          torch::TensorOptions().dtype(scalar_type).device(device));
      AllClose(res, xla_res);
    });
    ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::linspace", cpp_test::GetIgnoredCounters());
  };
}

TEST_F(AtenXlaTensorTest, TestLinspaceSmallSteps) {
  torch::Scalar start = 0;
  torch::Scalar end = 10;
  for (int64_t steps : {0, 1, 2}) {
    torch::Tensor res = torch::linspace(start, end, steps);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_res = torch::linspace(
          start, end, steps, torch::TensorOptions().device(device));
      AllClose(res, xla_res);
    });
    ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    ExpectCounterChanged("xla::linspace", cpp_test::GetIgnoredCounters());
  }
}

TEST_F(AtenXlaTensorTest, TestLinspaceReverse) {
  torch::Scalar start = 0;
  torch::Scalar end = -10;
  int64_t steps = 100;
  torch::Tensor res = torch::linspace(start, end, steps);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_res = torch::linspace(
        start, end, steps, torch::TensorOptions().device(device));
    AllClose(res, xla_res);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::linspace", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestNanToNum) {
  for (torch::ScalarType scalar_type :
       {torch::kHalf, torch::kFloat, torch::kDouble, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor input =
        isFloatingType(scalar_type)
            ? torch::tensor(
                  {1.0, std::nan("1"), std::numeric_limits<double>::infinity(),
                   -std::numeric_limits<double>::infinity()},
                  torch::TensorOptions(scalar_type))
            : torch::randint(0, 100, {3, 4}, torch::TensorOptions(scalar_type));
    torch::Tensor output = torch::nan_to_num(input);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::nan_to_num(xla_input);
      if (static_cast<XlaDeviceType>(
              bridge::AtenDeviceToXlaDevice(device).type()) ==
              XlaDeviceType::TPU &&
          scalar_type == torch::kDouble) {
        // Since TPU converts double to float (unlike CPU), the Inf entries are
        // expected to be different. Skipping checks for Inf entries.
        AllEqual(output[0], xla_output[0]);
        AllEqual(output[1], xla_output[1]);
      } else {
        AllClose(output, xla_output);
      }
    });
    output =
        torch::nan_to_num(input, /*nan=*/1.0, /*posinf=*/2.0, /*neginf=*/3.0);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::nan_to_num(
          xla_input, /*nan=*/1.0, /*posinf=*/2.0, /*neginf=*/3.0);
      AllClose(output, xla_output);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::nan_to_num", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestNanToNumOut) {
  for (torch::ScalarType scalar_type :
       {torch::kHalf, torch::kFloat, torch::kDouble, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor input =
        isFloatingType(scalar_type)
            ? torch::tensor(
                  {1.0, std::nan("1"), std::numeric_limits<double>::infinity(),
                   -std::numeric_limits<double>::infinity()},
                  torch::TensorOptions(scalar_type))
            : torch::randint(0, 100, {3, 4}, torch::TensorOptions(scalar_type));
    torch::Tensor output = torch::zeros_like(input);
    torch::nan_to_num_out(output, input);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::zeros_like(input);
      torch::nan_to_num_out(xla_output, xla_input);
      if (static_cast<XlaDeviceType>(
              bridge::AtenDeviceToXlaDevice(device).type()) ==
              XlaDeviceType::TPU &&
          scalar_type == torch::kDouble) {
        // Since TPU converts double to float (unlike CPU), the Inf entries are
        // expected to be different. Skipping checks for Inf entries.
        AllEqual(output[0], xla_output[0]);
        AllEqual(output[1], xla_output[1]);
      } else {
        AllClose(output, xla_output);
      }
    });
    torch::nan_to_num_out(output, input, /*nan=*/1.0, /*posinf=*/2.0,
                          /*neginf=*/3.0);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::zeros_like(input);
      torch::nan_to_num_out(xla_output, xla_input, /*nan=*/1.0, /*posinf=*/2.0,
                            /*neginf=*/3.0);
      AllClose(output, xla_output);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::nan_to_num", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestRoll) {
  std::vector<int64_t> input_shape = {2, 3, 4};
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor input =
        isFloatingType(scalar_type)
            ? torch::rand(input_shape, torch::TensorOptions(scalar_type))
            : torch::randint(0, 100, input_shape,
                             torch::TensorOptions(scalar_type));
    std::vector<std::vector<int64_t>> dim_powerset = {
        {}, {0}, {1}, {2}, {0, 1}, {1, 2}, {2, 0}, {0, 1, 2}};
    std::vector<std::vector<std::vector<int64_t>>> shift_set = {
        {{0}, {1}, {1}, {-24}, {24}, {-27}, {27}},
        {{0}, {-1}, {1}, {-5}, {5}},
        {{0}, {-1}, {1}, {-5}, {5}},
        {{0}, {-1}, {1}, {-5}, {5}},
        {{0, 0}, {-1, 4}},
        {{1, 2}, {0, -1}},
        {{0, 2}, {-1, 0}},
        {{4, 3, 2}, {-4, 3, 2}},
    };
    for (size_t i = 0; i < dim_powerset.size(); ++i) {
      std::vector<int64_t> roll_dims = dim_powerset[i];
      for (bool negative_dims : {false, true}) {
        if (negative_dims) {
          std::for_each(roll_dims.begin(), roll_dims.end(),
                        [](int64_t& dim) { dim -= 3; });
        }
        for (std::vector<int64_t> roll_shifts : shift_set[i]) {
          torch::Tensor output = torch::roll(input, roll_shifts, roll_dims);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output =
                torch::roll(xla_input, roll_shifts, roll_dims);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::roll", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestViewIsAliasOf) {
  torch::Tensor a = torch::empty(4, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::empty(4, torch::TensorOptions(torch::kFloat));

  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    EXPECT_EQ(!a.is_alias_of(b), !xla_a.is_alias_of(xla_b));

    torch::Tensor c = a.view({2, 2});
    torch::Tensor xla_c = xla_a.view({2, 2});
    EXPECT_EQ(a.is_alias_of(c), xla_a.is_alias_of(xla_c));

    torch::Tensor d = c.view({1, 4});
    torch::Tensor lazy_d = xla_c.view({1, 4});
    EXPECT_EQ(d.is_alias_of(c), lazy_d.is_alias_of(xla_c));
    EXPECT_EQ(d.is_alias_of(a), lazy_d.is_alias_of(xla_a));
  });
}

TEST_F(AtenXlaTensorTest, TestExpandIsAliasOf) {
  torch::Tensor a = torch::empty(4, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.expand(4, 3);
  EXPECT_TRUE(a.is_alias_of(b));

  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = xla_a.expand(4, 3);
    EXPECT_EQ(a.is_alias_of(b), xla_a.is_alias_of(xla_b));
  });
}

TEST_F(AtenXlaTensorTest, TestCdistForward) {
  torch::Tensor a =
      torch::rand({2, 20, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b =
      torch::rand({2, 10, 5}, torch::TensorOptions(torch::kFloat));
  std::vector<double> p_list = {0.0, 1.0, 2.0, 5.0,
                                std::numeric_limits<double>::infinity()};
  for (const auto& p : p_list) {
    torch::Tensor c = torch::cdist(a, b, p);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = torch::cdist(xla_a, xla_b, p);
      AllClose(c, xla_c);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::_cdist_forward", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenXlaTensorTest, TestGlu) {
  std::vector<std::vector<int64_t>> sizes{
      {3, 8}, {3, 5, 6}, {3, 8, 5}, {3, 8, 8, 16}};
  std::vector<int64_t> dims{-1, -1, 1, 3};

  auto size_it = sizes.begin();
  auto dim_it = dims.begin();
  for (; size_it != sizes.end() && dim_it != dims.end(); ++size_it, ++dim_it) {
    torch::Tensor input =
        torch::rand(*size_it, torch::TensorOptions(torch::kFloat));
    torch::Tensor output = torch::glu(input, *dim_it);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::glu(xla_input, *dim_it);
      AllClose(output, xla_output);
    });
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::glu", cpp_test::GetIgnoredCounters());
}

}  // namespace cpp_test
}  // namespace torch_xla
