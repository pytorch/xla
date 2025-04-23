#include "torch_xla/csrc/tf_image_resize_ops.h"

#include "torch_xla/csrc/shape_helper.h"
#include "xla/client/lib/constants.h"
#include "xla/tsl/lib/math/math_util.h"

namespace torch_xla::tf {

// We implement bilinear interpolation by upsampling followed by convolution.
// The basic idea is as follows. To scale from NxN to RxR:
//
//    1. S := (N - 1) /  gcd(N-1, R-1)
//    2. k := (R - 1) /  gcd(N-1, R-1)
//    3. Convolution((2k-1)x(2k-1), stride=k, lhs_dilation=S, padding=k-1)
//
// For example, to Scale from 7x7 -> 15x15:
//
//    1. S := (7-1) / gcd(7-1, 15-1) = 6 / gcd(6, 14) = 6 / 2 = 3
//    2. k := (15 - 1) / gcd(7-1, 15-1) = 14 / gcd(6, 14) = 14 / 2 = 7
//    3. Convolution(15x15, stride=3, lhs_dilation=7, padding=2)
//
//
// The 7x7 -> 15x15 case is much too large to write out in full as an
// example. The smallest interesting example is 3x3 -> 4x4.
//
// S := 2
// k := 3
//
// 00 03 06    00 00 00 00 00 00 00 00 00 00 00      00 02 04 06
// 09 12 15 -> 00 00 00 00 00 00 00 00 00 00 00   -> 06 08 10 12
// 18 21 24    00 00 00 00 00 03 00 00 06 00 00      12 14 16 18
//             00 00 00 00 00 00 00 00 00 00 00      18 20 22 24
//             00 00 00 00 00 00 00 00 00 00 00
//             00 00 09 00 00 12 00 00 15 00 00
//             00 00 00 00 00 00 00 00 00 00 00
//             00 00 00 00 00 00 00 00 00 00 00
//             00 00 18 00 00 21 00 00 24 00 00
//             00 00 00 00 00 00 00 00 00 00 00
//             00 00 00 00 00 00 00 00 00 00 00
//
// with the following convolutional kernel, with stride [2, 2]:
//       1 2 3 2 1
//       2 4 6 4 2
// 1/9 * 3 6 9 6 3
//       2 4 6 4 2
//       1 2 3 2 1
// Note that the convolution kernel matrix is separable and thus we can instead
// use 2 consecutive 1D kernel of the dimension 2k-1, along each axis.

// Computes the size of the convolutional kernel and stride to use when resizing
// from in_size to out_size.
struct ResizeConvolutionDims {
  // Size of the kernel to use.
  std::vector<int64_t> kernel_size;  // k

  // Stride of the convolution to use.
  std::vector<int64_t> stride;  // S
};

template <class T>
static void pv(const std::vector<T>& v, std::string s = "") {
    std::cout << s << " [ ";
    for (auto i : v) {
        std::cout << i << " ";
    }
    std::cout << "]" << std::endl;
}

ResizeConvolutionDims ComputeResizeConvolutionParameters(
    absl::Span<const int64_t> in_size, absl::Span<const int64_t> out_size,
    bool align_corners) {
  CHECK_EQ(in_size.size(), out_size.size());
  int num_spatial_dims = in_size.size();
  ResizeConvolutionDims dims;
  dims.kernel_size.resize(num_spatial_dims);
  dims.stride.resize(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    if (in_size[i] == 1) {
      // We must handle input size 1 specially because XLA convolution does
      // not allow stride 0.
      dims.stride[i] = dims.kernel_size[i] = 1;
    } else if (out_size[i] == 1) {
      // If in_size[i] > 1 but out_size[i] == 1, then we slice out the first
      // entry before resizing.
      dims.stride[i] = dims.kernel_size[i] = 1;
    } else {
      // The scaling factor changes depending on the alignment of corners.
      const int64_t in_size_factor =
          align_corners ? in_size[i] - 1 : in_size[i];
      const int64_t out_size_factor =
          align_corners ? out_size[i] - 1 : out_size[i];

      int64_t gcd = tsl::MathUtil::GCD(static_cast<uint64_t>(in_size_factor),
                                       static_cast<uint64_t>(out_size_factor));
      dims.stride[i] = in_size_factor / gcd;
      dims.kernel_size[i] = out_size_factor / gcd;
    }
  }
  pv(dims.kernel_size, "kernel_size:");
  pv(dims.stride, "stride:");
  return dims;
}

// Form a 2D convolution kernel like:
//       1 2 3 2 1
//       2 4 6 4 2
// 1/9 * 3 6 9 6 3
//       2 4 6 4 2
//       1 2 3 2 1
// by multiplying two 1D kernels of the form:
// 1/3 * [1 2 3 2 1]
// If the 2D kernel would be very large, the 1D kernel can be applied once in
// each dimension due to the symmetry of the kernel along all axis to reduce the
// computational intensity.
xla::XlaOp MakeBilinear1DKernel(xla::XlaBuilder* builder,
                                xla::PrimitiveType type, int64_t n) {
  std::vector<float> kernel(n * 2 - 1);
  for (int64_t i = 0; i < n; ++i) {
    float v = (i + 1.0f) / n;
    kernel[i] = v;
    kernel[n * 2 - 2 - i] = v;
  }
  return xla::ConvertElementType(xla::ConstantR1<float>(builder, kernel), type);
}

// Unlike the bilinear kernel, which is triangular, the nearest neighbor
// kernel is a square. For example, a 1D kernel with n=3 would look like
// [0 1 1 1 0]
// and n=4 would look like
// [0 0 1 1 1 1 0].
// Note that in the second case, the kernel is not symmetric and we default
// to the right (because an existing non TPU kernel
// for nearest neighbor resize already chose to default to the right,
// so we want to be consistent).
xla::XlaOp MakeNearestNeighbor1DKernel(xla::XlaBuilder* builder,
                                       xla::PrimitiveType type, int64_t n) {
  std::vector<float> kernel(n * 2 - 1, 0.0f);
  std::fill(&kernel[n / 2], &kernel[(3 * n) / 2], 1.0f);

  return xla::ConvertElementType(xla::ConstantR1<float>(builder, kernel), type);
}

// Kernels with more than 16 spatial elements are considered intense and the
// kernel should be applied to each dimension independently.
const int64_t kMax2DKernelSize = 16;

xla::XlaOp MakeGeneralResizeKernel(xla::XlaBuilder* builder,
                                   xla::PrimitiveType type,
                                   absl::Span<const int64_t> kernel_size,
                                   int64_t channels, bool is_kernel_bilinear) {
  auto make_kernel_func =
      is_kernel_bilinear ? MakeBilinear1DKernel : MakeNearestNeighbor1DKernel;

  std::vector<int64_t> depthwise_kernel_sizes = {
      (2 * kernel_size[0] - 1), (2 * kernel_size[1] - 1), channels, 1};
  auto depthwise_kernel =
      xla::BroadcastInDim(make_kernel_func(builder, type, kernel_size[1]),
                          depthwise_kernel_sizes, /*broadcast_dimensions=*/{1});

  return xla::Mul(depthwise_kernel,
                  make_kernel_func(builder, type, kernel_size[0]),
                  /*broadcast_dimensions=*/{0});
}

xla::XlaOp MakeGeneralResizeKernelInDim(xla::XlaBuilder* builder,
                                        xla::PrimitiveType type,
                                        absl::Span<const int64_t> kernel_size,
                                        int64_t channels, int64_t dim,
                                        bool is_kernel_bilinear) {
  auto make_kernel_func =
      is_kernel_bilinear ? MakeBilinear1DKernel : MakeNearestNeighbor1DKernel;

  std::vector<int64_t> depthwise_kernel_sizes = {
      dim == 0 ? (2 * kernel_size[0] - 1) : 1,
      dim == 1 ? (2 * kernel_size[1] - 1) : 1, channels, 1};
  return xla::BroadcastInDim(make_kernel_func(builder, type, kernel_size[dim]),
                             depthwise_kernel_sizes,
                             /*broadcast_dimensions=*/{dim});
}

xla::XlaOp BroadcastSpatialDimensions(xla::XlaBuilder* builder,
                                      const xla::XlaOp input,
                                      int32_t spatial_dimensions_offset,
                                      absl::Span<const int64_t> in_size,
                                      absl::Span<const int64_t> out_size) {
  // Add broadcasts to handle expanding from a size == 1 dimension to a
  // size > 1 dimension.
  auto broadcast_shape_or_status = builder->GetShape(input);
  if (!broadcast_shape_or_status.ok()) {
    return builder->ReportError(broadcast_shape_or_status.status());
  }
  xla::Shape broadcast_shape = broadcast_shape_or_status.value();
  for (int32_t i = 0; i < in_size.size(); ++i) {
    if (in_size[i] == 1 && out_size[i] > 1) {
      broadcast_shape.set_dimensions(spatial_dimensions_offset + i,
                                     out_size[i]);
    }
  }
  return xla::BroadcastInDim(input, broadcast_shape.dimensions(),
                             /*broadcast_dimensions=*/{0, 1, 2, 3});
}

xla::XlaOp ResizeUsingDilationAndConvolutionGradOp(
    xla::XlaBuilder* builder, const xla::XlaOp grad, xla::PrimitiveType type,
    const int num_spatial_dims, absl::Span<const int64_t> in_size,
    absl::Span<const int64_t> grad_size, const int64_t channels,
    const bool align_corners, bool is_kernel_bilinear) {
  ResizeConvolutionDims dims =
      ComputeResizeConvolutionParameters(in_size, grad_size, align_corners);

  // To form the backward convolution, we keep the kernel unchanged (it is
  // already symmetric) and swap the roles of strides and LHS dilation.
  xla::ConvolutionDimensionNumbers dimension_numbers;
  dimension_numbers.set_input_batch_dimension(0);
  dimension_numbers.set_output_batch_dimension(0);
  dimension_numbers.set_input_feature_dimension(num_spatial_dims + 1);
  dimension_numbers.set_output_feature_dimension(num_spatial_dims + 1);
  for (int i = 0; i < num_spatial_dims; ++i) {
    dimension_numbers.add_input_spatial_dimensions(i + 1);
    dimension_numbers.add_output_spatial_dimensions(i + 1);
    dimension_numbers.add_kernel_spatial_dimensions(i);
  }
  dimension_numbers.set_kernel_input_feature_dimension(num_spatial_dims + 1);
  dimension_numbers.set_kernel_output_feature_dimension(num_spatial_dims);
  xla::XlaOp output;
  if (dims.kernel_size[0] * dims.kernel_size[1] < kMax2DKernelSize) {
    std::cout << "dims.kernel_size[0] * dims.kernel_size[1] < kMax2DKernelSize :: " << dims.kernel_size[0] << " * " << dims.kernel_size[1] << " < " << kMax2DKernelSize << std::endl;
    xla::XlaOp kernel = MakeGeneralResizeKernel(builder, type, dims.kernel_size,
                                                channels, is_kernel_bilinear);

    // Broadcast the input kernel where the forward op expanded from a size == 1
    // dimension to a size > 1 dimension. This has the effect of summing the
    // gradient contributions in that dimension.
    kernel = BroadcastSpatialDimensions(
        builder, kernel, /*spatial_dimensions_offset=*/0, in_size, grad_size);

    output = xla::ConvGeneralDilated(
        grad, kernel, /*window_strides=*/dims.kernel_size,
        /*padding=*/
        {{dims.kernel_size[0] - 1, dims.kernel_size[0] - 1},
         {dims.kernel_size[1] - 1, dims.kernel_size[1] - 1}},
        /*lhs_dilation=*/dims.stride,
        /*rhs_dilation=*/{1, 1}, dimension_numbers,
        /*feature_group_count=*/channels);
  } else {
    std::cout << "dims.kernel_size[0] * dims.kernel_size[1] >= kMax2DKernelSize :: " << dims.kernel_size[0] << " * " << dims.kernel_size[1] << " >= " << kMax2DKernelSize << std::endl;
    xla::XlaOp kernel0 = MakeGeneralResizeKernelInDim(
        builder, type, dims.kernel_size, channels, 0, is_kernel_bilinear);
    xla::XlaOp kernel1 = MakeGeneralResizeKernelInDim(
        builder, type, dims.kernel_size, channels, 1, is_kernel_bilinear);

    // Broadcast the input kernel where the forward op expanded from a
    // size == 1 dimension to a size > 1 dimension. This has the effect of
    // summing the gradient contributions in that dimension.
    if (in_size[0] == 1 && grad_size[0] > 1) {
      kernel0 = BroadcastSpatialDimensions(builder, kernel0,
                                           /*spatial_dimensions_offset=*/0, {1},
                                           {grad_size[0]});
    }
    if (in_size[1] == 1 && grad_size[1] > 1) {
      kernel1 = BroadcastSpatialDimensions(builder, kernel0,
                                           /*spatial_dimensions_offset=*/0,
                                           in_size, grad_size);
    }

    output = xla::ConvGeneralDilated(
        grad, kernel0, /*window_strides=*/{dims.kernel_size[0], 1},
        /*padding=*/
        {{dims.kernel_size[0] - 1, dims.kernel_size[0] - 1}, {0, 0}},
        /*lhs_dilation=*/{dims.stride[0], 1},
        /*rhs_dilation=*/{1, 1}, dimension_numbers,
        /*feature_group_count=*/channels);

    output = xla::ConvGeneralDilated(
        output, kernel1, /*window_strides=*/{1, dims.kernel_size[1]},
        /*padding=*/
        {{0, 0}, {dims.kernel_size[1] - 1, dims.kernel_size[1] - 1}},
        /*lhs_dilation=*/{1, dims.stride[1]},
        /*rhs_dilation=*/{1, 1}, dimension_numbers,
        /*feature_group_count=*/channels);
  }

  std::cout << "Output: " << ShapeHelper::ShapeOfXlaOp(output).ToString() << std::endl;

  // If in_size[i] > 1 and grad_size[i] == 1, pad the output in dimension i.
  // Opposite of the slice performed by the forward op.
  xla::PaddingConfig padding = xla::MakeNoPaddingConfig(4);
  bool pad_output = false;
  for (int i = 0; i < num_spatial_dims; ++i) {
    if (in_size[i] > 1 && grad_size[i] == 1) {
      pad_output = true;
      padding.mutable_dimensions(1 + i)->set_edge_padding_high(in_size[i] - 1);
    }
  }
  if (pad_output) {
    output = xla::Pad(output, xla::Zero(builder, type), padding);
  }
  return output;
}

}  // namespace torch_xla::tf
