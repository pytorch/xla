#include "torch_xla/csrc/tensor_ops.h"

#include <ATen/core/Reduction.h>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

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
                xla::int64 dim) {
  if (dim < 0) {
    auto input_shape_ref = input.shape();
    auto dim_3_it = std::find((*input_shape_ref).dimensions().begin(),
                              (*input_shape_ref).dimensions().end(), 3);
    XLA_CHECK(dim_3_it != (*input_shape_ref).dimensions().end())
        << "No dimension of size 3 in input: " << (*input_shape_ref).ToString();
    dim = dim_3_it - (*input_shape_ref).dimensions().begin();
  }
  XLA_CHECK_EQ(input.size(dim), 3)
      << "Invalid cross argument: dimension " << dim << " does not have size 3";
  XLA_CHECK_LT(dim, input.shape().get().rank())
      << "Invalid cross argument: dimension " << dim << " out of range";
  // Extract the slices for each axis.
  XLATensor u1 = IndexAcrossDims(input, dim, 0);
  XLATensor v1 = IndexAcrossDims(other, dim, 0);
  XLATensor u2 = IndexAcrossDims(input, dim, 1);
  XLATensor v2 = IndexAcrossDims(other, dim, 1);
  XLATensor u3 = IndexAcrossDims(input, dim, 2);
  XLATensor v3 = IndexAcrossDims(other, dim, 2);
  // Compute the term for each axis.
  at::Scalar one(1);
  XLATensor s1 =
      XLATensor::sub(XLATensor::mul(u2, v3), XLATensor::mul(u3, v2), one);
  XLATensor s2 =
      XLATensor::sub(XLATensor::mul(u3, v1), XLATensor::mul(u1, v3), one);
  XLATensor s3 =
      XLATensor::sub(XLATensor::mul(u1, v2), XLATensor::mul(u2, v1), one);
  // Stack the terms into one result tensor.
  return XLATensor::stack({s1, s2, s3}, dim);
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

}  // namespace tensor_ops
}  // namespace torch_xla
