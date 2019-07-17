#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

struct BatchNormOutput {
  xla::XlaOp output;
  xla::XlaOp batch_mean;
  xla::XlaOp batch_variance;
};

struct BatchNormGrads {
  xla::XlaOp grad_input;
  xla::XlaOp grad_weight;
  xla::XlaOp grad_bias;
};

xla::XlaOp BatchNormVarianceInvert(const xla::XlaOp& variance, float eps_value);

BatchNormOutput BuildBatchNormTraining(const xla::XlaOp& input,
                                       const xla::XlaOp& weight,
                                       const xla::XlaOp& bias, float eps_value);

xla::XlaOp BuildBatchNormInference(const xla::XlaOp& input,
                                   const xla::XlaOp& weight,
                                   const xla::XlaOp& bias,
                                   const xla::XlaOp& mean,
                                   const xla::XlaOp& variance, float eps_value);

BatchNormGrads BuildBatchNormBackward(const xla::XlaOp& grad,
                                      const xla::XlaOp& input,
                                      const xla::XlaOp& weight,
                                      const xla::XlaOp& save_mean,
                                      const xla::XlaOp& save_invstd,
                                      bool training, float eps_value);

}  // namespace torch_xla
