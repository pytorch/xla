#ifndef XLA_TORCH_XLA_CSRC_SOFTMAX_BUILDER_H_
#define XLA_TORCH_XLA_CSRC_SOFTMAX_BUILDER_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

// Computes log(softmax(logits)) along the dimension specified by "dim".
xla::XlaOp BuildLogSoftmax(xla::XlaOp logits, int64_t dim);

// Computes the gradient of the input of the LogSoftmax function.
xla::XlaOp BuildLogSoftmaxGrad(xla::XlaOp grad_output, xla::XlaOp output,
                               int64_t dim);

xla::XlaOp BuildSoftmax(xla::XlaOp logits, int64_t dim);

xla::XlaOp BuildSoftmaxGrad(xla::XlaOp grad_output, xla::XlaOp output,
                            int64_t dim);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_SOFTMAX_BUILDER_H_