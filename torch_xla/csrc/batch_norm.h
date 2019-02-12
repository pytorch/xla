#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch/csrc/jit/ir.h"

namespace torch_xla {

struct BatchNormOutput {
  xla::XlaOp output;
  xla::XlaOp save_mean;        // batch_mean
  xla::XlaOp save_invstd_eps;  // 1 / sqrt(batch_var + eps)
};

struct BatchNormGrads {
  xla::XlaOp grad_input;
  xla::XlaOp grad_weight;
  xla::XlaOp grad_bias;
};

BatchNormOutput BuildBatchNorm(const xla::XlaOp& input,
                               const xla::XlaOp& weight, const xla::XlaOp& bias,
                               float eps_value);

BatchNormGrads BuildBatchNormBackward(const xla::XlaOp& grad,
                                      const xla::XlaOp& input,
                                      const xla::XlaOp& weight,
                                      const xla::XlaOp& save_mean,
                                      const xla::XlaOp& save_invstd_eps,
                                      float eps_value);

}  // namespace torch_xla
