#pragma once

#include <string>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch_xla {
namespace resize {

xla::Shape GetForwardOutputShape2d(
    const xla::Shape& input_shape,
    tensorflow::gtl::ArraySlice<const xla::int64> output_size);

xla::Shape GetBackwardOutputShape2d(
    const xla::Shape& input_shape,
    tensorflow::gtl::ArraySlice<const xla::int64> input_size);

xla::XlaOp LowerForward2d(const std::string& target, const xla::XlaOp& input,
                          const xla::Shape& output_shape, bool align_corners,
                          bool half_pixel_centers);

xla::XlaOp LowerBackward2d(const std::string& target, const xla::XlaOp& input,
                           const xla::Shape& output_shape, bool align_corners,
                           bool half_pixel_centers);

}  // namespace resize
}  // namespace torch_xla
