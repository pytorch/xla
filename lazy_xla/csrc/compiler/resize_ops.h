#pragma once

#include <string>

#include "absl/types/span.h"
#include "lazy_tensors/shape.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_lazy_tensors {
namespace resize {

lazy_tensors::Shape GetForwardOutputShape2d(
    const lazy_tensors::Shape& input_shape,
    absl::Span<const xla::int64> output_size);

lazy_tensors::Shape GetBackwardOutputShape2d(
    const lazy_tensors::Shape& input_shape,
    absl::Span<const xla::int64> input_size);

xla::XlaOp LowerForward2d(const std::string& target, xla::XlaOp input,
                          const xla::Shape& output_shape, bool align_corners,
                          bool half_pixel_centers);

xla::XlaOp LowerBackward2d(const std::string& target, xla::XlaOp input,
                           const xla::Shape& output_shape, bool align_corners,
                           bool half_pixel_centers);

}  // namespace resize
}  // namespace torch_lazy_tensors
