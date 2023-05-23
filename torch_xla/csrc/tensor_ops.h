#ifndef XLA_TORCH_XLA_CSRC_TENSOR_OPS_H_
#define XLA_TORCH_XLA_CSRC_TENSOR_OPS_H_

#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/tensor.h"

// Certain tensor operations can be expressed in terms of other tensor
// operations. Add their implementations here instead of the main XLATensorPtr
// class.

namespace torch_xla {
namespace tensor_ops {

XLATensorPtr Cross(const XLATensorPtr& input, const XLATensorPtr& other,
                   c10::optional<int64_t> dim);

XLATensorPtr MakeMatrixWithDiagonal(const XLATensorPtr& input,
                                    int64_t diagonal);

XLATensorPtr SmoothL1Loss(const XLATensorPtr& input, const XLATensorPtr& target,
                          ReductionMode reduction, double beta);

XLATensorPtr SmoothL1LossBackward(const XLATensorPtr& grad_output,
                                  const XLATensorPtr& input,
                                  const XLATensorPtr& target,
                                  ReductionMode reduction, double beta);

XLATensorPtr Softplus(const XLATensorPtr& input, const at::Scalar& beta,
                      const at::Scalar& threshold);

XLATensorPtr SoftplusBackward(const XLATensorPtr& grad_output,
                              const XLATensorPtr& input, const at::Scalar& beta,
                              const at::Scalar& threshold);

XLATensorPtr Select(const XLATensorPtr& input, int64_t dim, int64_t index);

XLATensorPtr EmbeddingDenseBackward(const XLATensorPtr& grad_output,
                                    const XLATensorPtr& indices,
                                    int64_t num_weights, int64_t padding_idx,
                                    bool scale_grad_by_freq);

}  // namespace tensor_ops
}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_TENSOR_OPS_H_