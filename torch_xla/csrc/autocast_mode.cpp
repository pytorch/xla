#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#include <ATen/autocast_mode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/library.h>

namespace at {
namespace autocast {
namespace {

#define KERNEL_XLA(OP, POLICY) KERNEL(c10::DeviceType::XLA, OP, POLICY)

#define KERNEL_XLA2(OP, OVERLOAD, POLICY) \
  KERNEL2(c10::DeviceType::XLA, OP, OVERLOAD, POLICY)

#define KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_XLA(                             \
    REDISPATCH_FUNC, REGISTER_NAME, REGISTER_SIGNATURE, REDISPATCH_SIGNATURE,  \
    POLICY)                                                                    \
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(c10::DeviceType::XLA, REDISPATCH_FUNC, \
                                        REGISTER_NAME, REGISTER_SIGNATURE,     \
                                        REDISPATCH_SIGNATURE, POLICY)

TORCH_LIBRARY_IMPL(_, AutocastXLA, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastXLA, m) {
  // lower_precision_fp cast policy
  KERNEL_XLA(conv1d, lower_precision_fp)
  KERNEL_XLA2(conv1d, padding, lower_precision_fp)
  KERNEL_XLA(conv2d, lower_precision_fp)
  KERNEL_XLA2(conv2d, padding, lower_precision_fp)
  KERNEL_XLA(conv3d, lower_precision_fp)
  KERNEL_XLA2(conv3d, padding, lower_precision_fp)
  KERNEL_XLA(bmm, lower_precision_fp)
  KERNEL_XLA(mm, lower_precision_fp)
  KERNEL_XLA(baddbmm, lower_precision_fp)
  KERNEL_XLA(addmm, lower_precision_fp)
  KERNEL_XLA(addbmm, lower_precision_fp)
  KERNEL_XLA(linear, lower_precision_fp)
  KERNEL_XLA(matmul, lower_precision_fp)
  KERNEL_XLA(conv_tbc, lower_precision_fp)
  KERNEL_XLA(conv_transpose1d, lower_precision_fp)
  KERNEL_XLA2(conv_transpose2d, input, lower_precision_fp)
  KERNEL_XLA2(conv_transpose3d, input, lower_precision_fp)
  KERNEL_XLA(prelu, lower_precision_fp)
  KERNEL_XLA(relu, lower_precision_fp)
  KERNEL_XLA(max_pool2d, lower_precision_fp)
  KERNEL_XLA(einsum, lower_precision_fp)
  // Disable `scaled_dot_product_attention` for now since it causes
  // undefined symbol with official torch whl.
  // KERNEL_XLA(scaled_dot_product_attention, lower_precision_fp)

  // fp32 cast policy
  // Commented out ops are included in the AutoCastCPU Policy,
  // but not lowered. Enable if op is lowered.
  KERNEL_XLA(batch_norm, fp32)
  KERNEL_XLA(_softmax, fp32)
  KERNEL_XLA2(softmax, int, fp32)
  KERNEL_XLA2(softmax, Dimname, fp32)
  KERNEL_XLA2(log_softmax, int, fp32)
  KERNEL_XLA2(log_softmax, Dimname, fp32)
  KERNEL_XLA(binary_cross_entropy, fp32)
  // KERNEL_XLA(grid_sampler, fp32)
  // KERNEL_XLA(polar, fp32)
  KERNEL_XLA(prod, fp32)
  KERNEL_XLA2(prod, dim_int, fp32)
  KERNEL_XLA2(prod, dim_Dimname, fp32)
  // KERNEL_XLA(quantile, fp32)
  // KERNEL_XLA2(quantile, scalar, fp32)
  // KERNEL_XLA(nanquantile, fp32)
  // KERNEL_XLA2(nanquantile, scalar, fp32)
  // KERNEL_XLA(stft, fp32)
  // KERNEL_XLA2(stft, center, fp32)
  KERNEL_XLA(cdist, fp32)
  // KERNEL_XLA(grid_sampler_2d, fp32)
  // KERNEL_XLA(grid_sampler_3d, fp32)
  KERNEL_XLA(trace, fp32)
  // KERNEL_XLA(view_as_complex, fp32)
  KERNEL_XLA(cholesky, fp32)
  KERNEL_XLA(cholesky_inverse, fp32)
  KERNEL_XLA(cholesky_solve, fp32)
  KERNEL_XLA(inverse, fp32)
  // KERNEL_XLA(lu_solve, fp32)
  // KERNEL_XLA(orgqr, fp32)
  // KERNEL_XLA(ormqr, fp32)
  // KERNEL_XLA(pinverse, fp32)
  KERNEL_XLA(reflection_pad1d, fp32)
  KERNEL_XLA(reflection_pad2d, fp32)
  KERNEL_XLA(replication_pad1d, fp32)
  KERNEL_XLA(replication_pad2d, fp32)
  KERNEL_XLA(replication_pad3d, fp32)
  KERNEL_XLA(mse_loss, fp32)
  KERNEL_XLA(cosine_embedding_loss, fp32)
  KERNEL_XLA(nll_loss, fp32)
  KERNEL_XLA(nll_loss2d, fp32)
  KERNEL_XLA(hinge_embedding_loss, fp32)
  // KERNEL_XLA(poisson_nll_loss, fp32)
  KERNEL_XLA(smooth_l1_loss, fp32)
  KERNEL_XLA(cross_entropy_loss, fp32)
  KERNEL_XLA(l1_loss, fp32)
  // KERNEL_XLA(huber_loss, fp32)
  KERNEL_XLA(margin_ranking_loss, fp32)
  KERNEL_XLA(soft_margin_loss, fp32)
  KERNEL_XLA(triplet_margin_loss, fp32)
  KERNEL_XLA(multi_margin_loss, fp32)
  KERNEL_XLA2(ctc_loss, IntList, fp32)
  KERNEL_XLA2(ctc_loss, Tensor, fp32)
  KERNEL_XLA(kl_div, fp32)
  KERNEL_XLA(multilabel_margin_loss, fp32)
  KERNEL_XLA(binary_cross_entropy_with_logits, fp32)
  // KERNEL_XLA(fft_fft, fp32)
  // KERNEL_XLA(fft_ifft, fp32)
  // KERNEL_XLA(fft_fft2, fp32)
  // KERNEL_XLA(fft_ifft2, fp32)
  // KERNEL_XLA(fft_fftn, fp32)
  // KERNEL_XLA(fft_ifftn, fp32)
  // KERNEL_XLA(fft_rfft, fp32)
  // KERNEL_XLA(fft_irfft, fp32)
  // KERNEL_XLA(fft_rfft2, fp32)
  // KERNEL_XLA(fft_irfft2, fp32)
  // KERNEL_XLA(fft_rfftn, fp32)
  // KERNEL_XLA(fft_irfftn, fp32)
  // KERNEL_XLA(fft_hfft, fp32)
  // KERNEL_XLA(fft_ihfft, fp32)
  // KERNEL_XLA(linalg_cond, fp32)
  // KERNEL_XLA2(linalg_cond, p_str, fp32)
  // KERNEL_XLA(linalg_matrix_rank, fp32)
  // KERNEL_XLA2(linalg_matrix_rank, tol_tensor, fp32)
  // KERNEL_XLA2(linalg_matrix_rank, atol_rtol_tensor, fp32)
  // KERNEL_XLA2(linalg_matrix_rank, atol_rtol_float, fp32)
  // KERNEL_XLA(linalg_solve, fp32)
  // KERNEL_XLA(linalg_cholesky, fp32)
  // KERNEL_XLA(linalg_svdvals, fp32)
  // KERNEL_XLA(linalg_eigvals, fp32)
  // KERNEL_XLA(linalg_eigvalsh, fp32)
  // KERNEL_XLA(linalg_inv, fp32)
  // KERNEL_XLA(linalg_householder_product, fp32)
  // KERNEL_XLA(linalg_tensorinv, fp32)
  // KERNEL_XLA(linalg_tensorsolve, fp32)
  // KERNEL_XLA(fake_quantize_per_tensor_affine, fp32)
  // KERNEL_XLA(geqrf, fp32)
  // KERNEL_XLA(_lu_with_info, fp32)
  KERNEL_XLA(qr, fp32)
  KERNEL_XLA(svd, fp32)
  KERNEL_XLA(triangular_solve, fp32)
  KERNEL_XLA(multilabel_margin_loss_forward, fp32)
  // KERNEL_XLA(linalg_qr, fp32)
  // KERNEL_XLA(linalg_cholesky_ex, fp32)
  KERNEL_XLA(linalg_svd, fp32)
  // KERNEL_XLA(linalg_eig, fp32)
  // KERNEL_XLA(linalg_eigh, fp32)
  // KERNEL_XLA(linalg_lstsq, fp32)
  KERNEL_XLA(linalg_inv_ex, fp32)

  // promote
  KERNEL_XLA(stack, promote)
  KERNEL_XLA(cat, promote)
  KERNEL_XLA(index_copy, promote)
  KERNEL_XLA2(index_copy, dimname, promote)
}

}  // namespace
}  // namespace autocast
}  // namespace at
