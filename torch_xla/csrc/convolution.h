#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch/csrc/jit/ir.h"

namespace torch_xla {

// Computes the convolution of the given input and kernel with the given
// precision, with the stride and padding specified by the node attributes.
xla::XlaOp BuildConvolution(const torch::jit::Node* node,
                            const xla::XlaOp& input, const xla::XlaOp& kernel);

// Same as above, with stride and padding provided as parameters.
xla::XlaOp BuildConvolution(
    const xla::XlaOp& input, const xla::XlaOp& kernel,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding);

// Same as above, then broadcasts the bias and adds it to the result.
xla::XlaOp BuildConvolutionBias(const torch::jit::Node* node,
                                const xla::XlaOp& input,
                                const xla::XlaOp& kernel,
                                const xla::XlaOp& bias);

// Same as above, with stride and padding provided as parameters.
xla::XlaOp BuildConvolutionBias(
    const xla::XlaOp& input, const xla::XlaOp& kernel, const xla::XlaOp& bias,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding);

xla::XlaOp BuildTransposedConvolution(
    const xla::XlaOp& input, const xla::XlaOp& kernel,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding);

xla::XlaOp BuildTransposedConvolutionBias(
    const xla::XlaOp& input, const xla::XlaOp& kernel, const xla::XlaOp& bias,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding);

struct Conv2DGrads {
  xla::XlaOp grad_input;
  xla::XlaOp grad_weight;
  xla::XlaOp grad_bias;
};

// Computes the gradients for a convolution.
Conv2DGrads BuildConv2dBackward(const torch::jit::Node* node,
                                const xla::XlaOp& grad_output,
                                const xla::XlaOp& input,
                                const xla::XlaOp& weight);

// Same as above, with stride and padding provided as parameters.
Conv2DGrads BuildConv2dBackward(
    const xla::XlaOp& grad_output, const xla::XlaOp& input,
    const xla::XlaOp& weight,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding);

Conv2DGrads BuildTransposedConvolutionBackward(
    const xla::XlaOp& grad_output, const xla::XlaOp& input,
    const xla::XlaOp& kernel,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding);

}  // namespace torch_xla
