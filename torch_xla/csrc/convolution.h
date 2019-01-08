#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch/csrc/jit/ir.h"

namespace torch_xla {

// Computes the convolution of the given input and kernel with the given
// precision, with the stride and padding specified by the node attributes.
xla::XlaOp BuildConvolution(
    const torch::jit::Node* node, const xla::XlaOp& input,
    const xla::XlaOp& kernel,
    const xla::PrecisionConfig::Precision conv_precision);

// Same as above, then broadcasts the bias and adds it to the result.
xla::XlaOp BuildConvolutionBias(
    const torch::jit::Node* node, const xla::XlaOp& input,
    const xla::XlaOp& kernel, const xla::XlaOp& bias,
    const xla::PrecisionConfig::Precision conv_precision);

struct Conv2DGrads {
  xla::XlaOp grad_input;
  xla::XlaOp grad_weight;
  xla::XlaOp grad_bias;
};

// Computes the gradients for a convolution.
Conv2DGrads BuildConv2dBackward(
    const torch::jit::Node* node, const xla::XlaOp& grad,
    const xla::XlaOp& input, const xla::XlaOp& weight,
    const xla::PrecisionConfig::Precision conv_precision);

}  // namespace torch_xla
