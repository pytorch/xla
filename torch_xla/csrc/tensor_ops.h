#pragma once

#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/tensor.h"

// Certain tensor operations can be expressed in terms of other tensor
// operations. Add their implementations here instead of the main XLATensor
// class.

namespace torch_xla {
namespace tensor_ops {

XLATensor Cross(const XLATensor& input, const XLATensor& other,
                c10::optional<xla::int64_t> dim);

XLATensor MakeMatrixWithDiagonal(const XLATensor& input, xla::int64_t diagonal);

XLATensor SmoothL1Loss(const XLATensor& input, const XLATensor& target,
                       ReductionMode reduction, double beta);

XLATensor SmoothL1LossBackward(const XLATensor& grad_output,
                               const XLATensor& input, const XLATensor& target,
                               ReductionMode reduction, double beta);

XLATensor Softplus(const XLATensor& input, const at::Scalar& beta,
                   const at::Scalar& threshold);

XLATensor SoftplusBackward(const XLATensor& grad_output, const XLATensor& input,
                           const at::Scalar& beta, const at::Scalar& threshold,
                           const XLATensor& output);

XLATensor Select(const XLATensor& input, xla::int64_t dim, xla::int64_t index);

XLATensor EmbeddingDenseBackward(const XLATensor& grad_output,
                                 const XLATensor& indices,
                                 xla::int64_t num_weights,
                                 xla::int64_t padding_idx,
                                 bool scale_grad_by_freq);

}  // namespace tensor_ops
}  // namespace torch_xla
