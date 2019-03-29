#include "torch_xla/csrc/tensor_ops.h"

#include <ATen/core/Reduction.h>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace tensor_ops {
namespace {

// Returns the sub-tensor at the given index in the given dimension. Its rank
// is one less than the input, in other words the singleton dimension is
// squeezed out.
XLATensor IndexAcrossDims(const XLATensor& input, xla::int64 dim,
                          xla::int64 index) {
  return XLATensor::squeeze(XLATensor::slice(input, dim, index, index + 1, 1));
}

}  // namespace

XLATensor Cross(const XLATensor& input, const XLATensor& other,
                c10::optional<xla::int64> dim) {
  xla::int64 canonical_dim;
  if (dim) {
    canonical_dim = XlaHelpers::GetCanonicalDimensionIndex(
        *dim, input.shape().get().rank());
  } else {
    auto input_shape_ref = input.shape();
    auto dim_3_it = std::find((*input_shape_ref).dimensions().begin(),
                              (*input_shape_ref).dimensions().end(), 3);
    XLA_CHECK(dim_3_it != (*input_shape_ref).dimensions().end())
        << "No dimension of size 3 in input: " << (*input_shape_ref).ToString();
    canonical_dim = dim_3_it - (*input_shape_ref).dimensions().begin();
  }
  XLA_CHECK_EQ(input.size(canonical_dim), 3)
      << "Invalid cross argument: dimension " << canonical_dim
      << " does not have size 3";
  XLA_CHECK_LT(canonical_dim, input.shape().get().rank())
      << "Invalid cross argument: dimension " << canonical_dim
      << " out of range";
  // Extract the slices for each axis.
  XLATensor u1 = IndexAcrossDims(input, canonical_dim, 0);
  XLATensor v1 = IndexAcrossDims(other, canonical_dim, 0);
  XLATensor u2 = IndexAcrossDims(input, canonical_dim, 1);
  XLATensor v2 = IndexAcrossDims(other, canonical_dim, 1);
  XLATensor u3 = IndexAcrossDims(input, canonical_dim, 2);
  XLATensor v3 = IndexAcrossDims(other, canonical_dim, 2);
  // Compute the term for each axis.
  at::Scalar one(1);
  XLATensor s1 =
      XLATensor::sub(XLATensor::mul(u2, v3), XLATensor::mul(u3, v2), one);
  XLATensor s2 =
      XLATensor::sub(XLATensor::mul(u3, v1), XLATensor::mul(u1, v3), one);
  XLATensor s3 =
      XLATensor::sub(XLATensor::mul(u1, v2), XLATensor::mul(u2, v1), one);
  // Stack the terms into one result tensor.
  return XLATensor::stack({s1, s2, s3}, canonical_dim);
}

XLATensor KlDivBackward(const XLATensor& grad_output, const XLATensor& input,
                        const XLATensor& target, xla::int64 reduction) {
  auto input_shape_ref = input.shape();
  XLATensor expanded_grad_output =
      XLATensor::expand(grad_output, input_shape_ref.get().dimensions());
  XLATensor grad_input = XLATensor::where(
      XLATensor::gt(target, 0),
      XLATensor::neg(XLATensor::mul(target, expanded_grad_output)),
      XLATensor::full_like(input, 0, input.GetDevice(), c10::nullopt));
  double input_elem_count = xla::ShapeUtil::ElementsIn(input_shape_ref.get());
  return reduction == Reduction::Mean
             ? XLATensor::div(grad_input, input_elem_count)
             : grad_input;
}

XLATensor MakeMatrixWithDiagonal(const XLATensor& input, xla::int64 diagonal) {
  xla::int64 size = input.shape().get().dimensions(0);
  XLATensor identity =
      XLATensor::eye(size, size, input.GetDevice(), input.dtype());
  auto padding = diagonal >= 0
                     ? std::vector<xla::int64>{diagonal, 0, 0, diagonal}
                     : std::vector<xla::int64>{0, -diagonal, -diagonal, 0};
  return XLATensor::constant_pad_nd(XLATensor::mul(identity, input), padding,
                                    0);
}

XLATensor SmoothL1Loss(const XLATensor& input, const XLATensor& target,
                       xla::int64 reduction) {
  auto broadcasted_inputs = XLATensor::broadcast_tensors({input, target});
  XLA_CHECK_EQ(broadcasted_inputs.size(), 2);
  const XLATensor& broadcasted_input = broadcasted_inputs[0];
  const XLATensor& broadcasted_target = broadcasted_inputs[1];
  at::Scalar one(1.);
  XLATensor diff = XLATensor::sub(broadcasted_input, broadcasted_target, one);
  at::Scalar half(0.5);
  XLATensor abs_diff = XLATensor::abs(diff);
  XLATensor squared_loss = XLATensor::mul(XLATensor::mul(diff, diff), half);
  XLATensor l1_loss = XLATensor::sub(abs_diff, half, one);
  XLATensor elementwise_loss =
      XLATensor::where(XLATensor::lt(abs_diff, one), squared_loss, l1_loss);
  auto all_dimensions =
      xla::util::Iota<xla::int64>((*broadcasted_input.shape()).rank());
  switch (reduction) {
    case Reduction::None:
      return elementwise_loss;
    case Reduction::Mean:
      return XLATensor::mean(elementwise_loss, all_dimensions, false,
                             broadcasted_input.dtype());
    case Reduction::Sum:
      return XLATensor::sum(elementwise_loss, all_dimensions, false,
                            broadcasted_input.dtype());
    default:
      XLA_ERROR() << "Invalid reduction type: " << reduction;
  }
}

XLATensor SmoothL1LossBackward(const XLATensor& grad_output,
                               const XLATensor& input, const XLATensor& target,
                               xla::int64 reduction) {
  auto broadcasted_inputs = XLATensor::broadcast_tensors({input, target});
  XLA_CHECK_EQ(broadcasted_inputs.size(), 2);
  const XLATensor& broadcasted_input = broadcasted_inputs[0];
  const XLATensor& broadcasted_target = broadcasted_inputs[1];
  at::Scalar one(1.);
  XLATensor diff = XLATensor::sub(broadcasted_input, broadcasted_target, one);
  XLATensor abs_diff = XLATensor::abs(diff);
  XLATensor grad_squared_loss =
      XLATensor::sub(broadcasted_input, broadcasted_target, one);
  XLATensor ones = XLATensor::full_like(broadcasted_input, one,
                                        broadcasted_input.GetDevice(),
                                        broadcasted_input.dtype());
  // NB: We can't use XLATensor::sign(), it returns zero for input zero.
  XLATensor grad_l1_loss =
      XLATensor::where(XLATensor::gt(broadcasted_input, broadcasted_target),
                       ones, XLATensor::neg(ones));
  XLATensor elementwise_loss_backward = XLATensor::where(
      XLATensor::lt(abs_diff, one), grad_squared_loss, grad_l1_loss);
  switch (reduction) {
    case Reduction::None:
    case Reduction::Sum:
      return XLATensor::mul(elementwise_loss_backward, grad_output);
    case Reduction::Mean: {
      double grad_scale = xla::ShapeUtil::ElementsIn(broadcasted_input.shape());
      return XLATensor::mul(
          XLATensor::div(elementwise_loss_backward, grad_scale), grad_output);
    }
    default:
      XLA_ERROR() << "Invalid reduction type: " << reduction;
  }
}

XLATensor Softplus(const XLATensor& input, at::Scalar beta,
                   at::Scalar threshold) {
  return XLATensor::where(
      XLATensor::gt(XLATensor::mul(input, beta), threshold), input,
      XLATensor::div(
          XLATensor::log1p(XLATensor::exp(XLATensor::mul(input, beta))), beta));
}

XLATensor SoftplusBackward(const XLATensor& grad_output, const XLATensor& input,
                           at::Scalar beta, at::Scalar threshold,
                           const XLATensor& output) {
  XLATensor scaled_output = XLATensor::mul(output, beta);
  XLATensor z = XLATensor::exp(scaled_output);
  return XLATensor::where(
      XLATensor::gt(scaled_output, threshold), grad_output,
      XLATensor::mul(grad_output, XLATensor::div(XLATensor::sub(z, 1, 1), z)));
}

XLATensor Select(const XLATensor& input, xla::int64 dim, xla::int64 index) {
  auto shape = input.shape();
  dim = XlaHelpers::GetCanonicalDimensionIndex(dim, shape.get().rank());
  XLATensor result = XLATensor::narrow(input, dim, index, 1);
  auto new_dims = XlaHelpers::DropDimensions(shape.get().dimensions(), {dim});
  return XLATensor::view(result, new_dims);
}

}  // namespace tensor_ops
}  // namespace torch_xla
