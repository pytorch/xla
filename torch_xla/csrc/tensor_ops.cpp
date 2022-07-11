#include "torch_xla/csrc/tensor_ops.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/lazy/core/helpers.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace tensor_ops {
namespace {

// Returns the sub-tensor at the given index in the given dimension. Its rank
// is one less than the input, in other words the singleton dimension is
// squeezed out.
XLATensorPtr IndexAcrossDims(const XLATensorPtr& input, int64_t dim,
                             int64_t index) {
  return XLATensor::squeeze(XLATensor::slice(input, dim, index, index + 1, 1),
                            dim);
}

}  // namespace

XLATensorPtr Cross(const XLATensorPtr& input, const XLATensorPtr& other,
                   c10::optional<int64_t> dim) {
  int64_t canonical_dim;
  if (dim) {
    canonical_dim = torch::lazy::GetCanonicalDimensionIndex(
        *dim, input->shape().get().rank());
  } else {
    auto input_shape_ref = input->shape();
    auto dim_3_it = std::find((*input_shape_ref).dimensions().begin(),
                              (*input_shape_ref).dimensions().end(), 3);
    XLA_CHECK(dim_3_it != (*input_shape_ref).dimensions().end())
        << "No dimension of size 3 in input: " << (*input_shape_ref).ToString();
    canonical_dim = dim_3_it - (*input_shape_ref).dimensions().begin();
  }
  XLA_CHECK_EQ(input->size(canonical_dim), 3)
      << "Invalid cross argument: dimension " << canonical_dim
      << " does not have size 3";
  XLA_CHECK_LT(canonical_dim, input->shape().get().rank())
      << "Invalid cross argument: dimension " << canonical_dim
      << " out of range";
  // Extract the slices for each axis.
  XLATensorPtr u1 = IndexAcrossDims(input, canonical_dim, 0);
  XLATensorPtr v1 = IndexAcrossDims(other, canonical_dim, 0);
  XLATensorPtr u2 = IndexAcrossDims(input, canonical_dim, 1);
  XLATensorPtr v2 = IndexAcrossDims(other, canonical_dim, 1);
  XLATensorPtr u3 = IndexAcrossDims(input, canonical_dim, 2);
  XLATensorPtr v3 = IndexAcrossDims(other, canonical_dim, 2);
  // Compute the term for each axis.
  at::Scalar one(1);
  XLATensorPtr s1 =
      XLATensor::sub(XLATensor::mul(u2, v3), XLATensor::mul(u3, v2), one);
  XLATensorPtr s2 =
      XLATensor::sub(XLATensor::mul(u3, v1), XLATensor::mul(u1, v3), one);
  XLATensorPtr s3 =
      XLATensor::sub(XLATensor::mul(u1, v2), XLATensor::mul(u2, v1), one);
  // Stack the terms into one result tensor.
  return XLATensor::stack({s1, s2, s3}, canonical_dim);
}

XLATensorPtr KlDivBackward(const XLATensorPtr& grad_output,
                           const XLATensorPtr& input,
                           const XLATensorPtr& target, ReductionMode reduction,
                           bool log_target) {
  auto input_shape_ref = input->shape();
  XLATensorPtr expanded_grad_output = XLATensor::expand(
      grad_output,
      torch::lazy::ToVector<int64_t>(input_shape_ref.get().dimensions()));
  XLATensorPtr grad_input;
  if (!log_target) {
    grad_input = XLATensor::where(
        XLATensor::gt(target, 0),
        XLATensor::neg(XLATensor::mul(target, expanded_grad_output)),
        XLATensor::full_like(input, 0, input->GetDevice(), c10::nullopt));
  } else {
    grad_input = XLATensor::neg(
        XLATensor::mul(XLATensor::exp(target), expanded_grad_output));
  }
  if (reduction == ReductionMode::kMean) {
    XLATensorPtr dims_size = XLATensor::get_dimensions_size(
        input, XlaHelpers::GetAllDimensions(input_shape_ref));
    grad_input = XLATensor::div(grad_input, dims_size);
  }
  return grad_input;
}

XLATensorPtr MakeMatrixWithDiagonal(const XLATensorPtr& input,
                                    int64_t diagonal) {
  int64_t size = input->shape().get().dimensions(0);
  XLATensorPtr identity =
      XLATensor::eye(size, size, input->GetDevice(), input->dtype());
  auto padding = diagonal >= 0
                     ? std::vector<int64_t>{diagonal, 0, 0, diagonal}
                     : std::vector<int64_t>{0, -diagonal, -diagonal, 0};
  return XLATensor::constant_pad_nd(XLATensor::mul(identity, input), padding,
                                    0);
}

XLATensorPtr SmoothL1Loss(const XLATensorPtr& input, const XLATensorPtr& target,
                          ReductionMode reduction, double beta) {
  torch::lazy::ScopePusher ir_scope(at::aten::smooth_l1_loss.toQualString());
  auto broadcasted_inputs = XLATensor::broadcast_tensors({input, target});
  XLA_CHECK_EQ(broadcasted_inputs.size(), 2);
  const XLATensorPtr& broadcasted_input = broadcasted_inputs[0];
  const XLATensorPtr& broadcasted_target = broadcasted_inputs[1];
  at::Scalar one(1.);
  at::Scalar beta_scalar(beta);
  XLATensorPtr diff =
      XLATensor::sub(broadcasted_input, broadcasted_target, one);
  at::Scalar half(0.5);
  at::Scalar half_beta(0.5 * beta);
  XLATensorPtr abs_diff = XLATensor::abs(diff);
  XLATensorPtr squared_loss = XLATensor::div(
      XLATensor::mul(XLATensor::mul(diff, diff), half), beta_scalar);
  XLATensorPtr l1_loss = XLATensor::sub(abs_diff, half_beta, one);
  XLATensorPtr elementwise_loss = XLATensor::where(
      XLATensor::lt(abs_diff, beta_scalar), squared_loss, l1_loss);
  auto all_dimensions =
      torch::lazy::Iota<int64_t>((*broadcasted_input->shape()).rank());
  switch (reduction) {
    case ReductionMode::kNone:
      return elementwise_loss;
    case ReductionMode::kMean:
      return XLATensor::mean(elementwise_loss, all_dimensions, false,
                             broadcasted_input->dtype());
    case ReductionMode::kSum:
      return XLATensor::sum(elementwise_loss, all_dimensions, false,
                            broadcasted_input->dtype());
    default:
      XLA_ERROR() << "Invalid reduction type: "
                  << torch::lazy::GetEnumValue(reduction);
  }
}

XLATensorPtr SmoothL1LossBackward(const XLATensorPtr& grad_output,
                                  const XLATensorPtr& input,
                                  const XLATensorPtr& target,
                                  ReductionMode reduction, double beta) {
  torch::lazy::ScopePusher ir_scope(
      at::aten::smooth_l1_loss_backward.toQualString());
  auto broadcasted_inputs = XLATensor::broadcast_tensors({input, target});
  XLA_CHECK_EQ(broadcasted_inputs.size(), 2);
  const XLATensorPtr& broadcasted_input = broadcasted_inputs[0];
  const XLATensorPtr& broadcasted_target = broadcasted_inputs[1];
  at::Scalar one(1.);
  at::Scalar beta_scalar(beta);
  XLATensorPtr diff =
      XLATensor::sub(broadcasted_input, broadcasted_target, one);
  XLATensorPtr abs_diff = XLATensor::abs(diff);
  XLATensorPtr grad_squared_loss = XLATensor::div(
      XLATensor::sub(broadcasted_input, broadcasted_target, one), beta_scalar);
  XLATensorPtr ones = XLATensor::full_like(broadcasted_input, one,
                                           broadcasted_input->GetDevice(),
                                           broadcasted_input->dtype());
  // NB: We can't use XLATensor::sign(), it returns zero for input zero.
  XLATensorPtr grad_l1_loss =
      XLATensor::where(XLATensor::gt(broadcasted_input, broadcasted_target),
                       ones, XLATensor::neg(ones));
  XLATensorPtr elementwise_loss_backward = XLATensor::where(
      XLATensor::lt(abs_diff, beta_scalar), grad_squared_loss, grad_l1_loss);
  switch (reduction) {
    case ReductionMode::kNone:
    case ReductionMode::kSum:
      return XLATensor::mul(elementwise_loss_backward, grad_output);
    case ReductionMode::kMean: {
      XLATensorPtr grad_scale = XLATensor::get_dimensions_size(
          broadcasted_input,
          XlaHelpers::GetAllDimensions(broadcasted_input->shape()));
      return XLATensor::mul(
          XLATensor::div(elementwise_loss_backward, grad_scale), grad_output);
    }
    default:
      XLA_ERROR() << "Invalid reduction type: "
                  << torch::lazy::GetEnumValue(reduction);
  }
}

XLATensorPtr Softplus(const XLATensorPtr& input, const at::Scalar& beta,
                      const at::Scalar& threshold) {
  return XLATensor::where(
      XLATensor::gt(XLATensor::mul(input, beta), threshold), input,
      XLATensor::div(
          XLATensor::log1p(XLATensor::exp(XLATensor::mul(input, beta))), beta));
}

XLATensorPtr SoftplusBackward(const XLATensorPtr& grad_output,
                              const XLATensorPtr& input, const at::Scalar& beta,
                              const at::Scalar& threshold) {
  XLATensorPtr scaled_input = XLATensor::mul(input, beta);
  XLATensorPtr z = XLATensor::exp(scaled_input);
  XLATensorPtr one_vec = XLATensor::full_like(z, 1, z->GetDevice(), z->dtype());

  return XLATensor::where(
      XLATensor::gt(scaled_input, threshold), grad_output,
      XLATensor::mul(grad_output,
                     XLATensor::div(z, XLATensor::add(z, one_vec, 1))));
}

XLATensorPtr Select(const XLATensorPtr& input, int64_t dim, int64_t index) {
  auto shape = input->shape();
  dim = torch::lazy::GetCanonicalDimensionIndex(dim, shape.get().rank());
  XLATensorPtr result = XLATensor::narrow(input, dim, index, 1);
  auto new_dims = torch::lazy::DropDimensions(
      xla::util::ToVector<int64_t>(shape.get().dimensions()),
      std::vector<int64_t>({dim}));
  return XLATensor::view(result, new_dims);
}

XLATensorPtr EmbeddingDenseBackward(const XLATensorPtr& grad_output,
                                    const XLATensorPtr& indices,
                                    int64_t num_weights, int64_t padding_idx,
                                    bool scale_grad_by_freq) {
  XLA_CHECK_EQ(indices->dtype(), at::ScalarType::Long)
      << "Embedding indices are expected to be of scalar type Long";
  auto indices_shape_ref = indices->shape();
  // The weight must be of rank 2, which means the rank of grad_output is one
  // more than the indices.
  XLA_CHECK_EQ(grad_output->shape().get().rank(),
               indices_shape_ref.get().rank() + 1);
  int64_t numel = xla::ShapeUtil::ElementsIn(indices_shape_ref.get());
  XLATensorPtr grad =
      XLATensor::view(grad_output, {numel, grad_output->size(-1)});
  XLATensorPtr grad_weight =
      XLATensor::full({num_weights, grad_output->size(-1)}, 0,
                      grad_output->GetDevice(), grad_output->dtype());
  XLATensorPtr indices_rank1 = XLATensor::view(indices, {numel});
  if (scale_grad_by_freq) {
    // Compute the histogram of index values.
    XLATensorPtr counts = XLATensor::full(
        {num_weights}, 0, indices->GetDevice(), indices->dtype());
    XLATensorPtr ones =
        XLATensor::full({numel}, 1, indices->GetDevice(), indices->dtype());
    XLATensor::index_put_(counts, counts, {indices_rank1}, /*start_dim=*/0,
                          /*values=*/ones,
                          /*accumulate=*/true, /*result_permutation=*/{0});
    XLATensorPtr grad_weights_scale =
        XLATensor::index(counts, {indices_rank1}, 0);
    // Scale the value of the gradient by the histogram.
    grad = XLATensor::div(grad, XLATensor::unsqueeze(grad_weights_scale, 1));
  }
  // Don't accumulate gradients for indices which are equal with the given
  // padding_idx.
  XLATensorPtr skip_padding = XLATensor::unsqueeze(
      XLATensor::ne(indices_rank1, static_cast<double>(padding_idx)), 1);
  skip_padding = XLATensor::expand(
      skip_padding,
      torch::lazy::ToVector<int64_t>(grad->shape().get().dimensions()));
  XLATensorPtr zero_grad =
      XLATensor::full_like(grad, 0, grad->GetDevice(), grad->dtype());
  return XLATensor::index_put(
      grad_weight, {indices_rank1},
      /*start_dim=*/0,
      /*values=*/XLATensor::where(skip_padding, grad, zero_grad),
      /*accumulate=*/true,
      /*result_permutation=*/{0, 1});
}

}  // namespace tensor_ops
}  // namespace torch_xla
