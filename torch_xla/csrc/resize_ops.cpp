#include "torch_xla/csrc/resize_ops.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace resize {
namespace {

std::string GetBackendConfig(bool align_corners, bool half_pixel_centers) {
  return absl::StrCat("\"", align_corners, half_pixel_centers, "\"");
}

double ResizeFactor(const xla::Shape& input_shape,
                    const xla::Shape& output_shape, int dim) {
  return static_cast<double>(input_shape.dimensions(dim)) /
         static_cast<double>(output_shape.dimensions(dim));
}

}  // namespace

xla::Shape GetForwardOutputShape2d(
    const xla::Shape& input_shape,
    tensorflow::gtl::ArraySlice<const xla::int64> output_size) {
  XLA_CHECK_EQ(output_size.size(), 2);
  return xla::ShapeUtil::MakeShape(
      input_shape.element_type(),
      {input_shape.dimensions(0), input_shape.dimensions(1), output_size[0],
       output_size[1]});
}

xla::Shape GetBackwardOutputShape2d(
    const xla::Shape& input_shape,
    tensorflow::gtl::ArraySlice<const xla::int64> input_size) {
  return xla::ShapeUtil::MakeShape(input_shape.element_type(), input_size);
}

xla::XlaOp LowerForward2d(const std::string& target, const xla::XlaOp& input,
                          const xla::Shape& output_shape, bool align_corners,
                          bool half_pixel_centers) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  if (input_shape.dimensions(2) == output_shape.dimensions(2) &&
      input_shape.dimensions(3) == output_shape.dimensions(3)) {
    return input;
  }
  if (input_shape.dimensions(2) == 1 && input_shape.dimensions(3) == 1) {
    return input + xla::Zeros(input.builder(), output_shape);
  }
  // XLA wants NHWC while PyTorch comes in as NCHW, so we need to transpose,
  // call the kernel, and transpose back.
  std::vector<xla::int64> transpose_permute({0, 3, 2, 1});
  auto inv_transpose_permute = xla::InversePermutation(transpose_permute);
  xla::Shape resized_shape =
      xla::ShapeUtil::PermuteDimensions(inv_transpose_permute, output_shape);
  xla::XlaOp tinput = xla::Transpose(input, transpose_permute);
  xla::XlaOp resised =
      xla::CustomCall(input.builder(), target, {tinput}, resized_shape,
                      GetBackendConfig(align_corners, half_pixel_centers));
  return xla::Transpose(resised, inv_transpose_permute);
}

xla::XlaOp LowerBackward2d(const std::string& target, const xla::XlaOp& input,
                           const xla::Shape& output_shape, bool align_corners,
                           bool half_pixel_centers) {
  static double resiple_split_factor =
      xla::sys_util::GetEnvDouble("XLA_RESIZE_SPLIT_FACTOR", 3.0);
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  if (input_shape.dimensions(2) == output_shape.dimensions(2) &&
      input_shape.dimensions(3) == output_shape.dimensions(3)) {
    return input;
  }
  // XLA wants NHWC while PyTorch comes in as NCHW, so we need to transpose,
  // call the kernel, and transpose back.
  std::vector<xla::int64> transpose_permute({0, 3, 2, 1});
  auto inv_transpose_permute = xla::InversePermutation(transpose_permute);
  xla::Shape resized_shape =
      xla::ShapeUtil::PermuteDimensions(inv_transpose_permute, output_shape);
  xla::XlaOp tinput = xla::Transpose(input, transpose_permute);
  std::string backend_config =
      GetBackendConfig(align_corners, half_pixel_centers);
  if (ResizeFactor(input_shape, output_shape, 2) > resiple_split_factor &&
      ResizeFactor(input_shape, output_shape, 3) > resiple_split_factor) {
    // If the resize is too large, do one dimension at a time.
    xla::Shape partial_shape = resized_shape;
    // Partial shape is in NHWC, while input shape is in NCHW.
    partial_shape.mutable_dimensions()[1] = input_shape.dimensions(2);
    tinput = xla::CustomCall(input.builder(), target, {tinput}, partial_shape,
                             backend_config);
  }
  xla::XlaOp resised = xla::CustomCall(input.builder(), target, {tinput},
                                       resized_shape, backend_config);
  return xla::Transpose(resised, inv_transpose_permute);
}

}  // namespace resize
}  // namespace torch_xla
