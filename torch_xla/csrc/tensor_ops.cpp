#include "torch_xla/csrc/tensor_ops.h"

#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/util.h>

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/util.h"
#include "torch_xla/csrc/tensor_methods.h"

namespace torch_xla {
namespace tensor_ops {
namespace {

// Returns the sub-tensor at the given index in the given dimension. Its rank
// is one less than the input, in other words the singleton dimension is
// squeezed out.
XLATensorPtr IndexAcrossDims(const XLATensorPtr& input, int64_t dim,
                             int64_t index) {
  return tensor_methods::squeeze(
      tensor_methods::slice(input, dim, index, index + 1, 1), dim);
}

}  // namespace

XLATensorPtr Cross(const XLATensorPtr& input, const XLATensorPtr& other,
                   std::optional<int64_t> dim) {
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
  XLATensorPtr s1 = tensor_methods::sub(tensor_methods::mul(u2, v3),
                                        tensor_methods::mul(u3, v2), one);
  XLATensorPtr s2 = tensor_methods::sub(tensor_methods::mul(u3, v1),
                                        tensor_methods::mul(u1, v3), one);
  XLATensorPtr s3 = tensor_methods::sub(tensor_methods::mul(u1, v2),
                                        tensor_methods::mul(u2, v1), one);
  // Stack the terms into one result tensor.
  return tensor_methods::stack({s1, s2, s3}, canonical_dim);
}

XLATensorPtr MakeMatrixWithDiagonal(const XLATensorPtr& input,
                                    int64_t diagonal) {
  int64_t size = input->shape().get().dimensions(0);
  XLATensorPtr identity =
      tensor_methods::eye(size, size, input->GetDevice(), input->dtype());
  auto padding = diagonal >= 0
                     ? std::vector<int64_t>{diagonal, 0, 0, diagonal}
                     : std::vector<int64_t>{0, -diagonal, -diagonal, 0};
  return tensor_methods::constant_pad_nd(tensor_methods::mul(identity, input),
                                         padding, 0);
}

XLATensorPtr SmoothL1Loss(const XLATensorPtr& input, const XLATensorPtr& target,
                          ReductionMode reduction, double beta) {
  torch::lazy::ScopePusher ir_scope(at::aten::smooth_l1_loss.toQualString());
  auto broadcasted_inputs = tensor_methods::broadcast_tensors({input, target});
  XLA_CHECK_EQ(broadcasted_inputs.size(), 2);
  const XLATensorPtr& broadcasted_input = broadcasted_inputs[0];
  const XLATensorPtr& broadcasted_target = broadcasted_inputs[1];
  at::Scalar one(1.);
  at::Scalar beta_scalar(beta);
  XLATensorPtr diff =
      tensor_methods::sub(broadcasted_input, broadcasted_target, one);
  at::Scalar half(0.5);
  at::Scalar half_beta(0.5 * beta);
  XLATensorPtr abs_diff = tensor_methods::abs(diff);
  XLATensorPtr squared_loss = tensor_methods::div(
      tensor_methods::mul(tensor_methods::mul(diff, diff), half), beta_scalar);
  XLATensorPtr l1_loss = tensor_methods::sub(abs_diff, half_beta, one);
  XLATensorPtr elementwise_loss = tensor_methods::where(
      tensor_methods::lt(abs_diff, beta_scalar), squared_loss, l1_loss);
  auto all_dimensions =
      torch::lazy::Iota<int64_t>((*broadcasted_input->shape()).rank());
  switch (reduction) {
    case ReductionMode::kNone:
      return elementwise_loss;
    case ReductionMode::kMean:
      return tensor_methods::mean(elementwise_loss, all_dimensions, false,
                                  broadcasted_input->dtype());
    case ReductionMode::kSum:
      return tensor_methods::sum(elementwise_loss, all_dimensions, false,
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
  auto broadcasted_inputs = tensor_methods::broadcast_tensors({input, target});
  XLA_CHECK_EQ(broadcasted_inputs.size(), 2);
  const XLATensorPtr& broadcasted_input = broadcasted_inputs[0];
  const XLATensorPtr& broadcasted_target = broadcasted_inputs[1];
  at::Scalar one(1.);
  at::Scalar beta_scalar(beta);
  XLATensorPtr diff =
      tensor_methods::sub(broadcasted_input, broadcasted_target, one);
  XLATensorPtr abs_diff = tensor_methods::abs(diff);
  XLATensorPtr grad_squared_loss = tensor_methods::div(
      tensor_methods::sub(broadcasted_input, broadcasted_target, one),
      beta_scalar);
  XLATensorPtr ones = tensor_methods::full_like(broadcasted_input, one,
                                                broadcasted_input->GetDevice(),
                                                broadcasted_input->dtype());
  // NB: We can't use tensor_methods::sign(), it returns zero for input zero.
  XLATensorPtr grad_l1_loss = tensor_methods::where(
      tensor_methods::gt(broadcasted_input, broadcasted_target), ones,
      tensor_methods::neg(ones));
  XLATensorPtr elementwise_loss_backward =
      tensor_methods::where(tensor_methods::lt(abs_diff, beta_scalar),
                            grad_squared_loss, grad_l1_loss);
  switch (reduction) {
    case ReductionMode::kNone:
    case ReductionMode::kSum:
      return tensor_methods::mul(elementwise_loss_backward, grad_output);
    case ReductionMode::kMean: {
      XLATensorPtr grad_scale = tensor_methods::get_dimensions_size(
          broadcasted_input,
          XlaHelpers::GetAllDimensions(broadcasted_input->shape()));
      return tensor_methods::mul(
          tensor_methods::div(elementwise_loss_backward, grad_scale),
          grad_output);
    }
    default:
      XLA_ERROR() << "Invalid reduction type: "
                  << torch::lazy::GetEnumValue(reduction);
  }
}

XLATensorPtr Softplus(const XLATensorPtr& input, const at::Scalar& beta,
                      const at::Scalar& threshold) {
  return tensor_methods::where(
      tensor_methods::gt(tensor_methods::mul(input, beta), threshold), input,
      tensor_methods::div(tensor_methods::log1p(tensor_methods::exp(
                              tensor_methods::mul(input, beta))),
                          beta));
}

XLATensorPtr SoftplusBackward(const XLATensorPtr& grad_output,
                              const XLATensorPtr& input, const at::Scalar& beta,
                              const at::Scalar& threshold) {
  XLATensorPtr scaled_input = tensor_methods::mul(input, beta);
  XLATensorPtr z = tensor_methods::exp(scaled_input);
  XLATensorPtr one_vec =
      tensor_methods::full_like(z, 1, z->GetDevice(), z->dtype());

  return tensor_methods::where(
      tensor_methods::gt(scaled_input, threshold), grad_output,
      tensor_methods::mul(
          grad_output,
          tensor_methods::div(z, tensor_methods::add(z, one_vec, 1))));
}

XLATensorPtr Select(const XLATensorPtr& input, int64_t dim, int64_t index) {
  auto shape = input->shape();
  dim = torch::lazy::GetCanonicalDimensionIndex(dim, shape.get().rank());
  XLATensorPtr result = tensor_methods::narrow(input, dim, index, 1);
  auto new_dims = torch::lazy::DropDimensions(
      torch_xla::runtime::util::ToVector<int64_t>(shape.get().dimensions()),
      std::vector<int64_t>({dim}));
  return tensor_methods::view(result, new_dims);
}

XLATensorPtr EmbeddingDenseBackward(const XLATensorPtr& grad_output,
                                    const XLATensorPtr& indices,
                                    int64_t num_weights, int64_t padding_idx,
                                    bool scale_grad_by_freq) {
  XLA_CHECK(indices->dtype() == at::ScalarType::Long ||
            indices->dtype() == at::ScalarType::Int);
  auto indices_shape_ref = indices->shape();
  // The weight must be of rank 2, which means the rank of grad_output is one
  // more than the indices.
  XLA_CHECK_EQ(grad_output->shape().get().rank(),
               indices_shape_ref.get().rank() + 1);
  int64_t numel = xla::ShapeUtil::ElementsIn(indices_shape_ref.get());
  XLATensorPtr grad =
      tensor_methods::view(grad_output, {numel, grad_output->size(-1)});
  XLATensorPtr grad_weight =
      tensor_methods::full({num_weights, grad_output->size(-1)}, 0,
                           grad_output->GetDevice(), grad_output->dtype());
  XLATensorPtr indices_rank1 = tensor_methods::view(indices, {numel});
  if (scale_grad_by_freq) {
    // Compute the histogram of index values.
    XLATensorPtr counts = tensor_methods::full(
        {num_weights}, 0, indices->GetDevice(), indices->dtype());
    XLATensorPtr ones = tensor_methods::full({numel}, 1, indices->GetDevice(),
                                             indices->dtype());
    tensor_methods::index_put_(counts, counts, {indices_rank1}, /*start_dim=*/0,
                               /*values=*/ones,
                               /*accumulate=*/true, /*result_permutation=*/{0});
    XLATensorPtr grad_weights_scale =
        tensor_methods::index(counts, {indices_rank1}, 0);
    // Scale the value of the gradient by the histogram.
    grad = tensor_methods::div(
        grad, tensor_methods::unsqueeze(grad_weights_scale, 1));
  }
  // Don't accumulate gradients for indices which are equal with the given
  // padding_idx.
  XLATensorPtr skip_padding = tensor_methods::unsqueeze(
      tensor_methods::ne(indices_rank1, static_cast<double>(padding_idx)), 1);
  skip_padding = tensor_methods::expand(
      skip_padding,
      torch::lazy::ToVector<int64_t>(grad->shape().get().dimensions()));
  XLATensorPtr zero_grad =
      tensor_methods::full_like(grad, 0, grad->GetDevice(), grad->dtype());
  return tensor_methods::index_put(
      grad_weight, {indices_rank1},
      /*start_dim=*/0,
      /*values=*/tensor_methods::where(skip_padding, grad, zero_grad),
      /*accumulate=*/true,
      /*result_permutation=*/{0, 1});
}

XLATensorPtr Embedding(const XLATensorPtr& weight,
                       const XLATensorPtr& indices) {
  XLA_CHECK_EQ(weight->shape().get().rank(), 2);
  XLA_CHECK(indices->dtype() == at::ScalarType::Long ||
            indices->dtype() == at::ScalarType::Int);

  if (indices->shape().get().rank() == 1) {
    return tensor_methods::index_select(weight, 0, indices);
  }

  std::vector<int64_t> final_size;
  int64_t num_elements = 1;
  for (int i = 0; i < indices->shape().get().rank(); i++) {
    int64_t dim = indices->shape().get().dimensions(i);
    final_size.push_back(dim);
    num_elements *= dim;
  }

  final_size.push_back(weight->shape().get().dimensions(1));

  XLATensorPtr embeddings = tensor_methods::index_select(
      weight, 0, tensor_methods::view(indices, {num_elements}));
  return tensor_methods::view(embeddings, final_size);
}

}  // namespace tensor_ops
}  // namespace torch_xla
