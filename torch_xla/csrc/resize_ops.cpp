#include "torch_xla/csrc/resize_ops.h"

#include "absl/strings/str_cat.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/shape_builder.h"
#include "torch_xla/csrc/shape_helper.h"
#include "xla/client/lib/constants.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace torch_xla {
namespace resize {
namespace {

xla::XlaOp BuildResize(xla::XlaOp input, const xla::Shape& output_shape,
                       bool align_corners, bool half_pixel_centers,
                       bool is_kernel_bilinear) {
  // Code copied from
  // https://github.com/tensorflow/tensorflow/blob/e51d6ab5730092775d516b18fa4ee85d49602cd8/tensorflow/compiler/tf2xla/kernels/image_resize_ops.cc#L477-L672
  //
  // Changes:
  // - Remove F32 data-type conversion when is_kernel_bilinear
  //   See: https://github.com/pytorch/xla/issues/7095

  // We implement bilinear interpolation and nearest neighbor with a Gather op.
  // For each output pixel, we gather the necessary slices of the input.
  // We then construct the weights that are necessary to calculate the weighted
  // sum for each output pixel. We do this with a DotGeneral op.
  xla::XlaBuilder* builder = input.builder();
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  XLA_CHECK_EQ(input_shape.rank(), 4)
      << "input must be 4-dimensional, got " << input_shape;

  // First dimension always assumed to be batch
  const int64_t batch = input_shape.dimensions(0);
  std::vector<int64_t> in_size = {input_shape.dimensions(1),
                                  input_shape.dimensions(2)};

  // Last/4th dimension always assumed to be num channels
  const int64_t channels = input_shape.dimensions(3);
  XLA_CHECK(in_size[0] > 0 && in_size[1] > 0) << absl::StrCat(
      "input size must be positive, got [", in_size[0], ",", in_size[1], "]");

  std::vector<int64_t> out_size = {output_shape.dimensions(1),
                                   output_shape.dimensions(2)};
  XLA_CHECK(out_size[0] > 0 && out_size[1] > 0)
      << absl::StrCat("output size must be positive, got [", out_size[0], ",",
                      out_size[1], "]");

  xla::PrimitiveType input_type = input_shape.element_type();
  xla::PrimitiveType output_type = output_shape.element_type();
  XLA_CHECK(input_type == output_type)
      << "input and output must have the same element type";

  xla::PrimitiveType original_input_type = input_type;
  if (xla::primitive_util::IsIntegralType(input_type)) {
    input = xla::ConvertElementType(input, xla::F32);
    input_type = xla::F32;
  }

  xla::XlaOp scalar_one_op =
      xla::ConvertElementType(xla::ConstantR0(builder, 1), input_type);
  xla::XlaOp scalar_half_op =
      xla::ConvertElementType(xla::ConstantR0(builder, 0.5), input_type);
  xla::XlaOp scalar_zero_op =
      xla::ConvertElementType(xla::ConstantR0(builder, 0), input_type);
  float h_scale;
  if (align_corners && out_size[0] > 1) {
    h_scale = (in_size[0] - 1) / static_cast<float>(out_size[0] - 1);
  } else {
    h_scale = in_size[0] / static_cast<float>(out_size[0]);
  }
  xla::XlaOp h_span_start = xla::Iota(
      builder, xla::ShapeUtil::MakeShape(input_type, {out_size[0]}), 0);
  if (half_pixel_centers) {
    h_span_start = xla::Add(h_span_start, scalar_half_op);
  }
  xla::XlaOp h_scale_op =
      xla::ConvertElementType(xla::ConstantR0(builder, h_scale), input_type);
  xla::XlaOp h_sample_f = xla::Mul(h_span_start, h_scale_op);

  if (is_kernel_bilinear) {
    h_span_start = xla::Sub(h_sample_f, scalar_one_op);
    if (half_pixel_centers) {
      h_span_start = xla::Sub(h_span_start, scalar_half_op);
    }
    h_span_start = xla::Ceil(h_span_start);
  } else {
    h_span_start =
        align_corners ? xla::Round(h_sample_f) : xla::Floor(h_sample_f);
  }
  const int64_t h_span_size =
      is_kernel_bilinear ? std::min(static_cast<int64_t>(3), in_size[0]) : 1;
  xla::XlaOp h_upper_bound = xla::ConvertElementType(
      xla::ConstantR0(builder, in_size[0] - h_span_size), input_type);
  if (!is_kernel_bilinear && !half_pixel_centers) {
    h_span_start = xla::Min(h_span_start, h_upper_bound);
  } else {
    h_span_start = xla::Clamp(scalar_zero_op, h_span_start, h_upper_bound);
  }
  xla::XlaOp broadcasted_h_span_start =
      xla::BroadcastInDim(h_span_start, {out_size[0], out_size[1], 1}, {0});

  float w_scale;
  if (align_corners && out_size[1] > 1) {
    w_scale = (in_size[1] - 1) / static_cast<float>(out_size[1] - 1);
  } else {
    w_scale = in_size[1] / static_cast<float>(out_size[1]);
  }
  xla::XlaOp w_span_start = xla::Iota(
      builder, xla::ShapeUtil::MakeShape(input_type, {out_size[1]}), 0);
  if (half_pixel_centers) {
    w_span_start = xla::Add(w_span_start, scalar_half_op);
  }
  xla::XlaOp w_scale_op =
      xla::ConvertElementType(xla::ConstantR0(builder, w_scale), input_type);
  xla::XlaOp w_sample_f = xla::Mul(w_span_start, w_scale_op);
  if (is_kernel_bilinear) {
    w_span_start = xla::Sub(w_sample_f, scalar_one_op);
    if (half_pixel_centers) {
      w_span_start = xla::Sub(w_span_start, scalar_half_op);
    }
    w_span_start = xla::Ceil(w_span_start);
  } else {
    w_span_start =
        align_corners ? xla::Round(w_sample_f) : xla::Floor(w_sample_f);
  }
  const int64_t w_span_size =
      is_kernel_bilinear ? std::min(static_cast<int64_t>(3), in_size[1]) : 1;
  xla::XlaOp w_upper_bound = xla::ConvertElementType(
      xla::ConstantR0(builder, in_size[1] - w_span_size), input_type);
  if (!is_kernel_bilinear && !half_pixel_centers) {
    w_span_start = xla::Min(w_span_start, w_upper_bound);
  } else {
    w_span_start = xla::Clamp(scalar_zero_op, w_span_start, w_upper_bound);
  }
  xla::XlaOp broadcasted_w_span_start =
      xla::BroadcastInDim(w_span_start, {out_size[0], out_size[1], 1}, {1});

  xla::XlaOp concatted = xla::ConvertElementType(
      xla::ConcatInDim(builder,
                       {broadcasted_h_span_start, broadcasted_w_span_start}, 2),
      xla::S32);

  absl::InlinedVector<int64_t, 4> slize_sizes = {batch, h_span_size,
                                                 w_span_size, channels};
  xla::GatherDimensionNumbers dimension_numbers;
  dimension_numbers.add_offset_dims(0);
  dimension_numbers.add_offset_dims(1);
  dimension_numbers.add_offset_dims(2);
  dimension_numbers.add_offset_dims(3);
  dimension_numbers.add_start_index_map(1);
  dimension_numbers.add_start_index_map(2);
  dimension_numbers.set_index_vector_dim(2);
  input = xla::Gather(input, concatted, dimension_numbers, slize_sizes, false);

  xla::XlaOp w_weight;
  if (is_kernel_bilinear) {
    xla::XlaOp w_sub = xla::Sub(w_span_start, w_sample_f);
    w_sub = xla::BroadcastInDim(w_sub, {out_size[1], w_span_size}, {0});
    xla::XlaOp w_offset = xla::Iota(
        builder, xla::ShapeUtil::MakeShape(input_type, {w_span_size}), 0);
    xla::XlaOp w_kernel_pos = xla::Add(w_sub, w_offset, {1});
    if (half_pixel_centers) {
      w_kernel_pos = xla::Add(w_kernel_pos, scalar_half_op);
    }
    w_weight = xla::Max(scalar_zero_op,
                        xla::Sub(scalar_one_op, xla::Abs(w_kernel_pos)));
  } else {
    w_weight = xla::Broadcast(scalar_one_op, {out_size[1], w_span_size});
  }
  xla::XlaOp w_weight_sum =
      xla::Reduce(w_weight, scalar_zero_op,
                  XlaHelpers::CreateAddComputation(input_type), {1});
  w_weight = xla::Div(w_weight, w_weight_sum, {0});

  xla::XlaOp h_weight;
  if (is_kernel_bilinear) {
    xla::XlaOp h_sub = xla::Sub(h_span_start, h_sample_f);
    h_sub = xla::BroadcastInDim(h_sub, {out_size[0], h_span_size}, {0});
    xla::XlaOp h_offset = xla::Iota(
        builder, xla::ShapeUtil::MakeShape(input_type, {h_span_size}), 0);
    xla::XlaOp h_kernel_pos = xla::Add(h_sub, h_offset, {1});
    if (half_pixel_centers) {
      h_kernel_pos = xla::Add(h_kernel_pos, scalar_half_op);
    }
    h_weight = xla::Max(scalar_zero_op,
                        xla::Sub(scalar_one_op, xla::Abs(h_kernel_pos)));
  } else {
    h_weight = xla::Broadcast(scalar_one_op, {out_size[0], h_span_size});
  }
  xla::XlaOp h_weight_sum =
      xla::Reduce(h_weight, scalar_zero_op,
                  XlaHelpers::CreateAddComputation(input_type), {1});
  h_weight = xla::Div(h_weight, h_weight_sum, {0});

  xla::DotDimensionNumbers dot_dnum;
  dot_dnum.add_lhs_contracting_dimensions(3);
  dot_dnum.add_lhs_contracting_dimensions(1);
  dot_dnum.add_rhs_contracting_dimensions(1);
  dot_dnum.add_rhs_contracting_dimensions(2);
  dot_dnum.add_lhs_batch_dimensions(2);
  dot_dnum.add_lhs_batch_dimensions(0);
  dot_dnum.add_rhs_batch_dimensions(4);
  dot_dnum.add_rhs_batch_dimensions(5);
  input = xla::DotGeneral(
      xla::DotGeneral(w_weight, h_weight, xla::DotDimensionNumbers()), input,
      dot_dnum);

  absl::InlinedVector<int64_t, 4> perm = {2, 0, 1, 3};
  input = xla::Transpose(input, perm);

  if (original_input_type != input_type) {
    input = xla::ConvertElementType(input, original_input_type);
  }
  return input;
}

std::string GetBackendConfig(bool align_corners, bool half_pixel_centers) {
  return absl::StrCat("\"", align_corners, half_pixel_centers, "\"");
}

double ResizeFactor(const xla::Shape& input_shape,
                    const xla::Shape& output_shape, int dim) {
  return static_cast<double>(input_shape.dimensions(dim)) /
         static_cast<double>(output_shape.dimensions(dim));
}

}  // namespace

xla::Shape GetForwardOutputShape2d(const xla::Shape& input_shape,
                                   absl::Span<const int64_t> output_size) {
  XLA_CHECK_EQ(output_size.size(), 2);
  return ShapeBuilder(input_shape.element_type())
      .Add(input_shape, 0)
      .Add(input_shape, 1)
      .Add(output_size[0])
      .Add(output_size[1])
      .Build();
}

xla::Shape GetBackwardOutputShape2d(const xla::Shape& input_shape,
                                    absl::Span<const int64_t> input_size) {
  return xla::ShapeUtil::MakeShape(input_shape.element_type(), input_size);
}

xla::XlaOp LowerForward2d(const std::string& target, xla::XlaOp input,
                          const xla::Shape& output_shape, bool align_corners,
                          bool half_pixel_centers) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  if (input_shape.dimensions(2) == output_shape.dimensions(2) &&
      input_shape.dimensions(3) == output_shape.dimensions(3)) {
    return input;
  }
  if (input_shape.dimensions(2) == 1 && input_shape.dimensions(3) == 1) {
    return input + xla::Zeros(input.builder(), output_shape);
  }
  // XLA wants NHWC while PyTorch comes in as NCHW, so we need to transpose,
  // call the kernel, and transpose back.
  std::vector<int64_t> transpose_permute({0, 3, 2, 1});
  auto inv_transpose_permute = xla::InversePermutation(transpose_permute);
  xla::Shape resized_shape =
      xla::ShapeUtil::PermuteDimensions(transpose_permute, output_shape);
  xla::XlaOp tinput = xla::Transpose(input, transpose_permute);

  xla::XlaOp resized;

  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(bridge::GetCurrentDevice().type());
  if (CheckTpuDevice(hw_type) || CheckNeuronDevice(hw_type)) {
    // TPU uses custom call implementation
    resized =
        xla::CustomCall(input.builder(), target, {tinput}, resized_shape,
                        GetBackendConfig(align_corners, half_pixel_centers));
  } else {
    bool is_kernel_bilinear = false;
    if (target == "ResizeBilinear") {
      is_kernel_bilinear = true;
    } else if (target == "ResizeNearest") {
      is_kernel_bilinear = false;
    } else {
      XLA_ERROR() << "Resize kernel: " << target << " is not supported";
    }
    resized = BuildResize(tinput, resized_shape, align_corners,
                          half_pixel_centers, is_kernel_bilinear);
  }
  return xla::Transpose(resized, inv_transpose_permute);
}

xla::XlaOp LowerBackward2d(const std::string& target, xla::XlaOp input,
                           const xla::Shape& output_shape, bool align_corners,
                           bool half_pixel_centers) {
  static double resiple_split_factor =
      torch_xla::runtime::sys_util::GetEnvDouble("XLA_RESIZE_SPLIT_FACTOR",
                                                 3.0);
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  if (input_shape.dimensions(2) == output_shape.dimensions(2) &&
      input_shape.dimensions(3) == output_shape.dimensions(3)) {
    return input;
  }
  // XLA wants NHWC while PyTorch comes in as NCHW, so we need to transpose,
  // call the kernel, and transpose back.
  std::vector<int64_t> transpose_permute({0, 3, 2, 1});
  auto inv_transpose_permute = xla::InversePermutation(transpose_permute);
  xla::Shape resized_shape =
      xla::ShapeUtil::PermuteDimensions(transpose_permute, output_shape);
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
