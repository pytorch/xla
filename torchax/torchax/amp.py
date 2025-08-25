import contextlib
import enum
import torch
from torch.utils import _pytree as pytree


# enum class CastPolicy : uint8_t {
#   lower_precision_fp = 0, // Cast all inputs to lower_precision_fp before
#                           // running the op. Currently, lower_precision_fp is
#                           // fp16 for AutocastCUDA, and is defined by user
#                           // (default bf16) for AutocastCPU or other device.
#   fp32, // Cast all inputs to at::kFloat before running the op.
#   fp32_set_opt_dtype, // Treats functions (like softmax) that
#                       //  1. we'd like to run in fp32 and
#                       //  2. have a std::optional<ScalarType> arg that controls
#                       //  the output type.
#                       // fp32_set_opt_dtype wrappers' policy is: if the output
#                       // type is already set, don't touch it, otherwise, set
#                       // it to at::kFloat.
#   fp32_append_dtype, // Treats functions (like norm) that
#                      //  1. we'd like to run in fp32 and
#                      //  2. have some overloads that accept an output type and
#                      //  other overloads that don't.
#                      // fp32_append_dtype wrappers wrap the overloads that don't
#                      // have an output dtype.
#                      // The wrapper policy is:  append at::kFloat to the args,
#                      // and redispatch to the type-aware overload.
#   promote, // Run in the widest dtype among several args.
# };
class CastPolicy(enum.Enum):
  LOWER_PRECISION_FP = 0
  FP32 = 1
  FP32_SET_OPT_DTYPE = 2
  FP32_APPEND_DTYPE = 3
  PROMOTE = 4


def execute_policy(policy, args, kwargs, target_lower_fp):

  def is_float(a):
    return isinstance(a, torch.Tensor) and a.is_floating_point()
  match policy:
    case CastPolicy.LOWER_PRECISION_FP:
      return pytree.tree_map_only(is_float, lambda a: a.to(target_lower_fp),
                                  (args, kwargs))
    case CastPolicy.FP32:
      return pytree.tree_map_only(is_float, lambda a: a.to(torch.float32),
                                  (args, kwargs))
    case CastPolicy.PROMOTE:
      dtypes = set(a.dtype for a in args)
      widest = max((dtype.itemsize, dtype) for dtype in dtypes)[1]
      return pytree.tree_map_only(is_float, lambda a: a.to(widest),
                                  (args, kwargs))
    case _:
      raise AssertionError(f'Policy {policy} not implemented yet.')


@contextlib.contextmanager
def autocast(device, dtype=torch.bfloat16, env=None):
  del device
  if env is None:
    import torchax
    env = torchax.default_env()
  with env.override_property(autocast_dtype=dtype):
    yield


# https://github.com/pytorch/pytorch/blob/05faba40287cf7d8734da96cb2e904f39710bf29/aten/src/ATen/autocast_mode.cpp#L327
autocast_policy = {
    torch.ops.aten.conv1d.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.conv1d.padding:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.conv2d.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.conv2d.padding:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.conv3d.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.conv3d.padding:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.bmm.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.mm.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.linalg_vecdot.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.baddbmm.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.addmm.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten._addmm_activation.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.addbmm.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.linear.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten._convolution.deprecated:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.matmul.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.conv_tbc.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.mkldnn_rnn_layer.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.conv_transpose1d.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.conv_transpose2d.input:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.conv_transpose3d.input:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.prelu.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten.scaled_dot_product_attention.default:
        CastPolicy.LOWER_PRECISION_FP,
    torch.ops.aten._native_multi_head_attention.default:
        CastPolicy.LOWER_PRECISION_FP,

    # fp32 cast policy
    torch.ops.aten.avg_pool3d.default:
        CastPolicy.FP32,
    torch.ops.aten.binary_cross_entropy.default:
        CastPolicy.FP32,
    torch.ops.aten.grid_sampler.default:
        CastPolicy.FP32,
    torch.ops.aten.polar.default:
        CastPolicy.FP32,
    torch.ops.aten.prod.default:
        CastPolicy.FP32,
    torch.ops.aten.prod.dim_int:
        CastPolicy.FP32,
    torch.ops.aten.prod.dim_Dimname:
        CastPolicy.FP32,
    torch.ops.aten.quantile.default:
        CastPolicy.FP32,
    torch.ops.aten.quantile.scalar:
        CastPolicy.FP32,
    torch.ops.aten.nanquantile.default:
        CastPolicy.FP32,
    torch.ops.aten.nanquantile.scalar:
        CastPolicy.FP32,
    torch.ops.aten.stft.default:
        CastPolicy.FP32,
    torch.ops.aten.stft.center:
        CastPolicy.FP32,
    torch.ops.aten.cdist.default:
        CastPolicy.FP32,
    torch.ops.aten.grid_sampler_2d.default:
        CastPolicy.FP32,
    torch.ops.aten._grid_sampler_2d_cpu_fallback.default:
        CastPolicy.FP32,
    torch.ops.aten.grid_sampler_3d.default:
        CastPolicy.FP32,
    torch.ops.aten.trace.default:
        CastPolicy.FP32,
    torch.ops.aten.view_as_complex.default:
        CastPolicy.FP32,
    torch.ops.aten.cholesky.default:
        CastPolicy.FP32,
    torch.ops.aten.cholesky_inverse.default:
        CastPolicy.FP32,
    torch.ops.aten.cholesky_solve.default:
        CastPolicy.FP32,
    torch.ops.aten.inverse.default:
        CastPolicy.FP32,
    torch.ops.aten.lu_solve.default:
        CastPolicy.FP32,
    torch.ops.aten.orgqr.default:
        CastPolicy.FP32,
    torch.ops.aten.ormqr.default:
        CastPolicy.FP32,
    torch.ops.aten.pinverse.default:
        CastPolicy.FP32,
    torch.ops.aten.max_pool3d.default:
        CastPolicy.FP32,
    torch.ops.aten.max_unpool2d.default:
        CastPolicy.FP32,
    torch.ops.aten.max_unpool3d.default:
        CastPolicy.FP32,
    torch.ops.aten.adaptive_avg_pool3d.default:
        CastPolicy.FP32,
    torch.ops.aten.reflection_pad1d.default:
        CastPolicy.FP32,
    torch.ops.aten.reflection_pad2d.default:
        CastPolicy.FP32,
    torch.ops.aten.replication_pad1d.default:
        CastPolicy.FP32,
    torch.ops.aten.replication_pad2d.default:
        CastPolicy.FP32,
    torch.ops.aten.replication_pad3d.default:
        CastPolicy.FP32,
    torch.ops.aten.mse_loss.default:
        CastPolicy.FP32,
    torch.ops.aten.cosine_embedding_loss.default:
        CastPolicy.FP32,
    torch.ops.aten.nll_loss.default:
        CastPolicy.FP32,
    torch.ops.aten.nll_loss2d.default:
        CastPolicy.FP32,
    torch.ops.aten.hinge_embedding_loss.default:
        CastPolicy.FP32,
    torch.ops.aten.poisson_nll_loss.default:
        CastPolicy.FP32,
    torch.ops.aten.smooth_l1_loss.default:
        CastPolicy.FP32,
    torch.ops.aten.cross_entropy_loss.default:
        CastPolicy.FP32,
    torch.ops.aten.l1_loss.default:
        CastPolicy.FP32,
    torch.ops.aten.huber_loss.default:
        CastPolicy.FP32,
    torch.ops.aten.margin_ranking_loss.default:
        CastPolicy.FP32,
    torch.ops.aten.soft_margin_loss.default:
        CastPolicy.FP32,
    torch.ops.aten.triplet_margin_loss.default:
        CastPolicy.FP32,
    torch.ops.aten.multi_margin_loss.default:
        CastPolicy.FP32,
    torch.ops.aten.ctc_loss.IntList:
        CastPolicy.FP32,
    torch.ops.aten.ctc_loss.Tensor:
        CastPolicy.FP32,
    torch.ops.aten.kl_div.default:
        CastPolicy.FP32,
    torch.ops.aten.multilabel_margin_loss.default:
        CastPolicy.FP32,
    torch.ops.aten.binary_cross_entropy_with_logits.default:
        CastPolicy.FP32,
    torch.ops.aten.fft_fft.default:
        CastPolicy.FP32,
    torch.ops.aten.fft_ifft.default:
        CastPolicy.FP32,
    torch.ops.aten.fft_fft2.default:
        CastPolicy.FP32,
    torch.ops.aten.fft_ifft2.default:
        CastPolicy.FP32,
    torch.ops.aten.fft_fftn.default:
        CastPolicy.FP32,
    torch.ops.aten.fft_ifftn.default:
        CastPolicy.FP32,
    torch.ops.aten.fft_rfft.default:
        CastPolicy.FP32,
    torch.ops.aten.fft_irfft.default:
        CastPolicy.FP32,
    torch.ops.aten.fft_rfft2.default:
        CastPolicy.FP32,
    torch.ops.aten.fft_irfft2.default:
        CastPolicy.FP32,
    torch.ops.aten.fft_rfftn.default:
        CastPolicy.FP32,
    torch.ops.aten.fft_irfftn.default:
        CastPolicy.FP32,
    torch.ops.aten.fft_hfft.default:
        CastPolicy.FP32,
    torch.ops.aten.fft_ihfft.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_cond.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_cond.p_str:
        CastPolicy.FP32,
    torch.ops.aten.linalg_matrix_rank.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_matrix_rank.tol_tensor:
        CastPolicy.FP32,
    torch.ops.aten.linalg_matrix_rank.atol_rtol_tensor:
        CastPolicy.FP32,
    torch.ops.aten.linalg_matrix_rank.atol_rtol_float:
        CastPolicy.FP32,
    torch.ops.aten.linalg_solve.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_cholesky.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_svdvals.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_eigvals.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_eigvalsh.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_inv.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_householder_product.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_tensorinv.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_tensorsolve.default:
        CastPolicy.FP32,
    torch.ops.aten.fake_quantize_per_tensor_affine.default:
        CastPolicy.FP32,
    torch.ops.aten.geqrf.default:
        CastPolicy.FP32,
    torch.ops.aten._lu_with_info.default:
        CastPolicy.FP32,
    torch.ops.aten.qr.default:
        CastPolicy.FP32,
    torch.ops.aten.svd.default:
        CastPolicy.FP32,
    torch.ops.aten.triangular_solve.default:
        CastPolicy.FP32,
    torch.ops.aten.fractional_max_pool2d.default:
        CastPolicy.FP32,
    torch.ops.aten.fractional_max_pool3d.default:
        CastPolicy.FP32,
    torch.ops.aten.adaptive_max_pool3d.default:
        CastPolicy.FP32,
    torch.ops.aten.multilabel_margin_loss_forward.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_qr.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_cholesky_ex.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_svd.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_eig.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_eigh.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_lstsq.default:
        CastPolicy.FP32,
    torch.ops.aten.linalg_inv_ex.default:
        CastPolicy.FP32,

    # promote
    torch.ops.aten.stack.default:
        CastPolicy.PROMOTE,
    torch.ops.aten.cat.default:
        CastPolicy.PROMOTE,
    torch.ops.aten.index_copy.default:
        CastPolicy.PROMOTE,
    torch.ops.aten.index_copy.dimname:
        CastPolicy.PROMOTE,
}
