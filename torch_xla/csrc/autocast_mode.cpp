#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/NativeFunctions.h>
#include <ATen/autocast_mode.h>
#include <ATen/Operators.h>

#include <c10/util/intrusive_ptr.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

namespace at {
namespace autocast {

// Base template for WrapFunction_, which is specialized to contain a "call" method each CastPolicy
template<CastPolicy policy, DeviceType device_type, class Redispatch, Redispatch* F, class Ret, class ArgList> struct WrapFunction_ {};

// CastPolicy::lower_precision_fp General_DeviceType
template<DeviceType device_type, class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::lower_precision_fp, device_type, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(get_autocast_dispatch_key_from_device_type(device_type));
    return (*F)(cached_cast(get_lower_precision_fp_from_device_type(device_type), args, device_type)...);
  }
};

// CastPolicy::fp32 General_DeviceType
template<DeviceType device_type, class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp32, device_type, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(get_autocast_dispatch_key_from_device_type(device_type));
    return (*F)(cached_cast(at::kFloat, args, device_type)...);
  }
};

// CastPolicy::promote General_DeviceType
template<DeviceType device_type, class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::promote, device_type, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(get_autocast_dispatch_key_from_device_type(device_type));
    auto to_type = promote_type(get_lower_precision_fp_from_device_type(device_type), device_type, args...);
    return (*F)(cached_cast(to_type, args, device_type)...);
  }
};

// Wrapper to infer return_type and parameter_types for WrapFunction_ (imitating pytorch/core/boxing/impl/WrapFunctionIntoFunctor.h)
template<CastPolicy policy,
         DeviceType device_type,
         class Registered, // The signature for which we're registering.  The dispatcher's calling code invokes our
                           // registered functions with arguments matching Registered, so we register
                           // WrapFunction_::call methods with a matching signature to properly field those arguments.
                           // guts::function_traits below extracts return_type and parameter_types from Registered,
                           // which WrapFunction_ templates above use to declare their call methods.
         class Redispatch, // The signature for the function we're redispatching to.  In most cases this is the same
                           // as Registered, but for some ops (for example, ops where we append a dtype) it's useful
                           // to redispatch to a function with a different signature.
         Redispatch* F>    // The actual function we're redispatching to.
struct WrapFunction final {
  using type = WrapFunction_<policy,
                             device_type,
                             Redispatch,
                             F,
                             typename guts::function_traits<Registered>::return_type,
                             typename guts::function_traits<Registered>::parameter_types>;
};

namespace {

// KERNEL_XLA registration for AutocastXLA
#define KERNEL_XLA(OP, POLICY) \
  m.impl(TORCH_SELECTIVE_NAME("aten::" #OP), \
    &WrapFunction<CastPolicy::POLICY, DeviceType::XLA, decltype(ATEN_FN(OP)), decltype(ATEN_FN(OP)), &ATEN_FN(OP)>::type::call);
#define KERNEL_XLA2(OP, OVERLOAD, POLICY) \
  m.impl(TORCH_SELECTIVE_NAME("aten::" #OP "." #OVERLOAD), \
    &WrapFunction<CastPolicy::POLICY, DeviceType::XLA, decltype(ATEN_FN2(OP, OVERLOAD)), decltype(ATEN_FN2(OP, OVERLOAD)), &ATEN_FN2(OP, OVERLOAD)>::type::call);

    
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

  // fp32 cast policy
  KERNEL_XLA(batch_norm, fp32)
  KERNEL_XLA2(log_softmax, int, fp32)
  KERNEL_XLA2(log_softmax, Dimname, fp32)
  KERNEL_XLA(binary_cross_entropy, fp32)
  KERNEL_XLA(grid_sampler, fp32)
  KERNEL_XLA(polar, fp32)
  KERNEL_XLA(prod, fp32)
  KERNEL_XLA2(prod, dim_int, fp32)
  KERNEL_XLA2(prod, dim_Dimname, fp32)
  KERNEL_XLA(quantile, fp32)
  KERNEL_XLA2(quantile, scalar, fp32)
  KERNEL_XLA(nanquantile, fp32)
  KERNEL_XLA2(nanquantile, scalar, fp32)
  KERNEL_XLA(stft, fp32)
  KERNEL_XLA2(stft, center, fp32)
  KERNEL_XLA(cdist, fp32)
  KERNEL_XLA(grid_sampler_2d, fp32)
  KERNEL_XLA(grid_sampler_3d, fp32)
  KERNEL_XLA(trace, fp32)
  KERNEL_XLA(view_as_complex, fp32)
  KERNEL_XLA(cholesky, fp32)
  KERNEL_XLA(cholesky_inverse, fp32)
  KERNEL_XLA(cholesky_solve, fp32)
  KERNEL_XLA(inverse, fp32)
  KERNEL_XLA(lu_solve, fp32)
  KERNEL_XLA(orgqr, fp32)
  KERNEL_XLA(ormqr, fp32)
  KERNEL_XLA(pinverse, fp32)
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
  KERNEL_XLA(poisson_nll_loss, fp32)
  KERNEL_XLA(smooth_l1_loss, fp32)
  KERNEL_XLA(cross_entropy_loss, fp32)
  KERNEL_XLA(l1_loss, fp32)
  KERNEL_XLA(huber_loss, fp32)
  KERNEL_XLA(margin_ranking_loss, fp32)
  KERNEL_XLA(soft_margin_loss, fp32)
  KERNEL_XLA(triplet_margin_loss, fp32)
  KERNEL_XLA(multi_margin_loss, fp32)
  KERNEL_XLA2(ctc_loss, IntList, fp32)
  KERNEL_XLA2(ctc_loss, Tensor, fp32)
  KERNEL_XLA(kl_div, fp32)
  KERNEL_XLA(multilabel_margin_loss, fp32)
  KERNEL_XLA(binary_cross_entropy_with_logits, fp32)
  KERNEL_XLA(fft_fft, fp32)
  KERNEL_XLA(fft_ifft, fp32)
  KERNEL_XLA(fft_fft2, fp32)
  KERNEL_XLA(fft_ifft2, fp32)
  KERNEL_XLA(fft_fftn, fp32)
  KERNEL_XLA(fft_ifftn, fp32)
  KERNEL_XLA(fft_rfft, fp32)
  KERNEL_XLA(fft_irfft, fp32)
  KERNEL_XLA(fft_rfft2, fp32)
  KERNEL_XLA(fft_irfft2, fp32)
  KERNEL_XLA(fft_rfftn, fp32)
  KERNEL_XLA(fft_irfftn, fp32)
  KERNEL_XLA(fft_hfft, fp32)
  KERNEL_XLA(fft_ihfft, fp32)
  KERNEL_XLA(linalg_cond, fp32)
  KERNEL_XLA2(linalg_cond, p_str, fp32)
  KERNEL_XLA(linalg_matrix_rank, fp32)
  KERNEL_XLA2(linalg_matrix_rank, tol_tensor, fp32)
  KERNEL_XLA2(linalg_matrix_rank, atol_rtol_tensor, fp32)
  KERNEL_XLA2(linalg_matrix_rank, atol_rtol_float, fp32)
  KERNEL_XLA(linalg_solve, fp32)
  KERNEL_XLA(linalg_cholesky, fp32)
  KERNEL_XLA(linalg_svdvals, fp32)
  KERNEL_XLA(linalg_eigvals, fp32)
  KERNEL_XLA(linalg_eigvalsh, fp32)
  KERNEL_XLA(linalg_inv, fp32)
  KERNEL_XLA(linalg_householder_product, fp32)
  KERNEL_XLA(linalg_tensorinv, fp32)
  KERNEL_XLA(linalg_tensorsolve, fp32)
  KERNEL_XLA(fake_quantize_per_tensor_affine, fp32)
  KERNEL_XLA(geqrf, fp32)
  KERNEL_XLA(_lu_with_info, fp32)
  KERNEL_XLA(qr, fp32)
  KERNEL_XLA(svd, fp32)
  KERNEL_XLA(triangular_solve, fp32)
  KERNEL_XLA(multilabel_margin_loss_forward, fp32)
  KERNEL_XLA(linalg_qr, fp32)
  KERNEL_XLA(linalg_cholesky_ex, fp32)
  KERNEL_XLA(linalg_svd, fp32)
  KERNEL_XLA(linalg_eig, fp32)
  KERNEL_XLA(linalg_eigh, fp32)
  KERNEL_XLA(linalg_lstsq, fp32)
  KERNEL_XLA(linalg_inv_ex, fp32)

  // promote
  KERNEL_XLA(stack, promote)
  KERNEL_XLA(cat, promote)
  KERNEL_XLA(index_copy, promote)
  KERNEL_XLA2(index_copy, dimname, promote)
}

} // namespace
} // namespace autocast
} // namespace at
