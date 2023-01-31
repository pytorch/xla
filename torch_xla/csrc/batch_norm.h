#pragma once

#include "xla/client/xla_builder.h"

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

xla::XlaOp BatchNormVarianceInvert(xla::XlaOp variance, float eps_value);

BatchNormOutput BuildBatchNormTraining(xla::XlaOp input, xla::XlaOp weight,
                                       xla::XlaOp bias, float eps_value);

xla::XlaOp BuildBatchNormInference(xla::XlaOp input, xla::XlaOp weight,
                                   xla::XlaOp bias, xla::XlaOp mean,
                                   xla::XlaOp variance, float eps_value);

BatchNormGrads BuildBatchNormBackward(xla::XlaOp grad, xla::XlaOp input,
                                      xla::XlaOp weight, xla::XlaOp save_mean,
                                      xla::XlaOp save_invstd, bool training,
                                      float eps_value);

}  // namespace torch_xla
