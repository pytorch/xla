#pragma once

#include "torch_xla/csrc/tensor.h"

// Certain tensor operations can be expressed in terms of other tensor
// operations. Add their implementations here instead of the main XLATensor
// class.

namespace torch_xla {
namespace tensor_ops {

XLATensor Cross(const XLATensor& input, const XLATensor& other,
                c10::optional<xla::int64> dim);

XLATensor KlDivBackward(const XLATensor& grad_output, const XLATensor& input,
                        const XLATensor& target, xla::int64 reduction);

XLATensor MakeMatrixWithDiagonal(const XLATensor& input, xla::int64 diagonal);

XLATensor SmoothL1Loss(const XLATensor& input, const XLATensor& target,
                       xla::int64 reduction);

XLATensor SmoothL1LossBackward(const XLATensor& grad_output,
                               const XLATensor& input, const XLATensor& target,
                               xla::int64 reduction);

XLATensor Softplus(const XLATensor& input, at::Scalar beta,
                   at::Scalar threshold);

XLATensor SoftplusBackward(const XLATensor& grad_output, const XLATensor& input,
                           at::Scalar beta, at::Scalar threshold,
                           const XLATensor& output);

XLATensor Select(const XLATensor& input, xla::int64 dim, xla::int64 index);

XLATensor EmbeddingDenseBackward(const XLATensor& grad_output,
                                 const XLATensor& indices,
                                 xla::int64 num_weights, xla::int64 padding_idx,
                                 bool scale_grad_by_freq);

}  // namespace tensor_ops
}  // namespace torch_xla
