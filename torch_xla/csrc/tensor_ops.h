#pragma once

#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/tensor.h"

// Certain tensor operations can be expressed in terms of other tensor
// operations. Add their implementations here instead of the main XLATensor
// class.

namespace torch_xla {
namespace tensor_ops {

XLATensor Cross(const XLATensor& input, const XLATensor& other,
                c10::optional<int64_t> dim);

XLATensor KlDivBackward(const XLATensor& grad_output, const XLATensor& input,
                        const XLATensor& target, ReductionMode reduction,
                        bool log_target);

XLATensor MakeMatrixWithDiagonal(const XLATensor& input, int64_t diagonal);

XLATensor SmoothL1Loss(const XLATensor& input, const XLATensor& target,
                       ReductionMode reduction, double beta);

XLATensor SmoothL1LossBackward(const XLATensor& grad_output,
                               const XLATensor& input, const XLATensor& target,
                               ReductionMode reduction, double beta);

XLATensor Softplus(const XLATensor& input, const at::Scalar& beta,
                   const at::Scalar& threshold);

XLATensor SoftplusBackward(const XLATensor& grad_output, const XLATensor& input,
                           const at::Scalar& beta, const at::Scalar& threshold);

XLATensor Select(const XLATensor& input, int64_t dim, int64_t index);

XLATensor EmbeddingDenseBackward(const XLATensor& grad_output,
                                 const XLATensor& indices,
                                 int64_t num_weights,
                                 int64_t padding_idx,
                                 bool scale_grad_by_freq);

}  // namespace tensor_ops
}  // namespace torch_xla
