#ifndef XLA_TORCH_XLA_CSRC_ELEMENTWISE_H_
#define XLA_TORCH_XLA_CSRC_ELEMENTWISE_H_

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

xla::XlaOp BuildPrelu(xla::XlaOp input, xla::XlaOp weight);

std::vector<xla::XlaOp> BuildRrelu(xla::XlaOp input, const at::Scalar& lower,
                                   const at::Scalar& upper, bool training,
                                   xla::XlaOp rng_seed);

xla::XlaOp BuildRreluBackward(xla::XlaOp grad_output, xla::XlaOp input,
                              xla::XlaOp noise, const at::Scalar& lower,
                              const at::Scalar& upper, bool training);

xla::XlaOp BuildHardshrink(xla::XlaOp input, xla::XlaOp lambda);
xla::XlaOp BuildHardSigmoid(xla::XlaOp input);
xla::XlaOp BuildHardSigmoidBackward(xla::XlaOp grad_output, xla::XlaOp input);
xla::XlaOp BuildHardSwish(xla::XlaOp input);
xla::XlaOp BuildHardSwishBackward(xla::XlaOp grad_output, xla::XlaOp input);
xla::XlaOp BuildSoftshrink(xla::XlaOp input, xla::XlaOp lambda);
xla::XlaOp BuildShrinkBackward(xla::XlaOp grad_output, xla::XlaOp input,
                               xla::XlaOp lambda);

xla::XlaOp BuildHardtanhBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                 const at::Scalar& min_val,
                                 const at::Scalar& max_val);

// Computes the leaky rectified linear unit:
// LeakyReLU(x) = max(0, input) + negative_slope ∗ min(0, input).
xla::XlaOp BuildLeakyRelu(xla::XlaOp input, xla::XlaOp negative_slope);

xla::XlaOp BuildLeakyReluBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                  xla::XlaOp negative_slope);

// Computes the sigmoid function using Tanh
// Sigmoid(x) = (tanh(x ∗ 0.5) + 1) ∗ 0.5
xla::XlaOp BuildSigmoid(xla::XlaOp input);

// Computes the backward of Silu
// grad_output * (sigmoid(input) * (1 + input * (1 - sigmoid(input))))
xla::XlaOp BuildSiLUBackward(xla::XlaOp grad_output, xla::XlaOp input);

// Computes the reciprocal function.
// Reciprocal(x) = 1 / x
xla::XlaOp BuildReciprocal(xla::XlaOp input);

// Computes the sgn of the complex input.
// If input magnitude is 0 then 0, else input / input magnitude
xla::XlaOp BuildSgn(xla::XlaOp input);

// Computes the sign of the input.
// If x is NaN then 0, otherwise the actual sign
xla::XlaOp BuildSign(xla::XlaOp input);

// Computes the absolute value of the input.
xla::XlaOp BuildAbs(xla::XlaOp input);

xla::XlaOp BuildSoftplus(xla::XlaOp input, xla::XlaOp beta,
                         xla::XlaOp threshold);

// Computes the GELU function of input.
// GELU(x) = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
xla::XlaOp BuildGelu(xla::XlaOp input);

// Computes the backward of GELU.
xla::XlaOp BuildGeluBackward(xla::XlaOp grad_output, xla::XlaOp input);

// Computes the CELU function of input.
// CELU(x)=max(0,x)+min(0,a*(exp(x/a)−1))
xla::XlaOp BuildCelu(xla::XlaOp input, const at::Scalar& alpha);

// Computes the SELU function of input.
// SELU(x)=scale*(max(0,x)+min(0,a*(exp(x)−1)))
xla::XlaOp BuildSelu(xla::XlaOp input);

// Computes the LogSigmoid function of input.
std::vector<xla::XlaOp> BuildLogSigmoid(xla::XlaOp input);

// Computes the backward of LogSigmoid.
xla::XlaOp BuildLogSigmoidBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                   xla::XlaOp buffer);

// Computes the Elu function of input.
xla::XlaOp BuildElu(xla::XlaOp input, xla::XlaOp alpha, xla::XlaOp scale,
                    xla::XlaOp input_scale);

// Computes the backward of Elu.
xla::XlaOp BuildEluBackward(xla::XlaOp grad_output, xla::XlaOp output,
                            const at::Scalar& alpha, const at::Scalar& scale,
                            const at::Scalar& input_scale);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_ELEMENTWISE_H_