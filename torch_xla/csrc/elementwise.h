#pragma once

#include <ATen/core/interned_strings.h>
#include <c10/core/Scalar.h>

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

// Computes binary comparison operations.
xla::XlaOp BuildComparisonOp(c10::Symbol kind, xla::XlaOp lhs, xla::XlaOp rhs);

// Computes the elementwise threshold of the input: if the value is below the
// threshold, replace it with the provided value, otherwise leave it unchanged.
xla::XlaOp BuildThreshold(xla::XlaOp input, xla::XlaOp output,
                          const float threshold, const float value);

// Computes the rectified linear unit (replace negative elements with 0).
xla::XlaOp BuildRelu(xla::XlaOp input);

std::vector<xla::XlaOp> BuildRrelu(xla::XlaOp input, at::Scalar lower,
                                   at::Scalar upper, bool training,
                                   xla::XlaOp rng_seed);

xla::XlaOp BuildRreluBackward(xla::XlaOp grad_output, xla::XlaOp input,
                              xla::XlaOp noise, at::Scalar lower,
                              at::Scalar upper, bool training);

xla::XlaOp BuildHardshrink(xla::XlaOp input, at::Scalar lambda);
xla::XlaOp BuildHardSigmoid(xla::XlaOp input);
xla::XlaOp BuildHardSigmoidBackward(xla::XlaOp grad_output, xla::XlaOp input);
xla::XlaOp BuildSoftshrink(xla::XlaOp input, at::Scalar lambda);
xla::XlaOp BuildShrinkBackward(xla::XlaOp grad_output, xla::XlaOp input,
                               at::Scalar lambda);

xla::XlaOp BuildHardtanhBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                 at::Scalar min_val, at::Scalar max_val);

// Computes the leaky rectified linear unit:
// LeakyReLU(x) = max(0, input) + negative_slope ∗ min(0, input).
xla::XlaOp BuildLeakyRelu(xla::XlaOp input, double negative_slope);

xla::XlaOp BuildLeakyReluBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                  double negative_slope_value);

// Computes the sigmoid function using Tanh
// Sigmoid(x) = (tanh(x ∗ 0.5) + 1) ∗ 0.5
xla::XlaOp BuildSigmoid(xla::XlaOp input);

// Computes the reciprocal function.
// Reciprocal(x) = 1 / x
xla::XlaOp BuildReciprocal(xla::XlaOp input);

// Computes the sign of the input.
// If x is NaN then 0, otherwise the actual sign
xla::XlaOp BuildSign(xla::XlaOp input);

// Computes the absolute value of the input.
xla::XlaOp BuildAbs(xla::XlaOp input);

}  // namespace torch_xla
