#ifndef XLA_TORCH_XLA_CSRC_TF_IMAGE_RESIZE_OPS_H_
#define XLA_TORCH_XLA_CSRC_TF_IMAGE_RESIZE_OPS_H_

#include "absl/types/span.h"
#include "xla/client/xla_builder.h"

namespace torch_xla::tf {

xla::XlaOp ResizeUsingDilationAndConvolutionGradOp(
    xla::XlaBuilder* builder, const xla::XlaOp grad, xla::PrimitiveType type,
    const int num_spatial_dims, absl::Span<const int64_t> in_size,
    absl::Span<const int64_t> grad_size, const int64_t channels,
    const bool align_corners, bool is_kernel_bilinear);

}  // namespace torch_xla::tf

#endif  // XLA_TORCH_XLA_CSRC_TF_IMAGE_RESIZE_OPS_H_
