#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

// Computes log(softmax(logits)) along the dimension specified by "dim".
xla::XlaOp BuildLogSoftmax(const xla::XlaOp& logits, xla::int64 dim);

// Computes the gradient of the input of the LogSoftmax function.
xla::XlaOp BuildLogSoftmaxGrad(const xla::XlaOp& grad_output,
                               const xla::XlaOp& output, xla::int64 dim);

xla::XlaOp BuildSoftmax(const xla::XlaOp& logits, xla::int64 dim);

xla::XlaOp BuildSoftmaxGrad(const xla::XlaOp& grad_output,
                            const xla::XlaOp& output, xla::int64 dim);

}  // namespace torch_xla
