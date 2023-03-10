#ifndef XLA_TORCH_XLA_CSRC_RESIZE_OPS_H_
#define XLA_TORCH_XLA_CSRC_RESIZE_OPS_H_

#include <string>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {
namespace resize {

xla::Shape GetForwardOutputShape2d(const xla::Shape& input_shape,
                                   absl::Span<const int64_t> output_size);

xla::Shape GetBackwardOutputShape2d(const xla::Shape& input_shape,
                                    absl::Span<const int64_t> input_size);

xla::XlaOp LowerForward2d(const std::string& target, xla::XlaOp input,
                          const xla::Shape& output_shape, bool align_corners,
                          bool half_pixel_centers);

xla::XlaOp LowerBackward2d(const std::string& target, xla::XlaOp input,
                           const xla::Shape& output_shape, bool align_corners,
                           bool half_pixel_centers);

}  // namespace resize
}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_RESIZE_OPS_H_