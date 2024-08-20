import unittest

import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, ops)
from torch.utils import _pytree as pytree
from torch_xla2 import tensor


skiplist = {
    "__rpow__",  # NOTE: cannot fix because torch test case has undefined behavior
                 # such as 0 to negative power.
    "_segment_reduce",
    "_upsample_bilinear2d_aa",
    "bincount", # NOTE: dtype for int input torch gives float. This is weird.
    "block_diag",
    "broadcast_tensors",
    "bucketize",
    "byte",
    "cat",
    "cauchy",
    "cdist",
    "cholesky",
    "cholesky_inverse",
    "cholesky_solve",
    "combinations",
    "complex",
    "count_nonzero",
    "cov",
    "cross",
    "cummax",
    "cummin",
    "cumsum",
    "diag",
    "diag_embed",
    "diagflat",
    "diagonal_copy",
    "diagonal_scatter",
    "diff",
    "digamma",
    "dist",
    "equal",
    "erfc",
    "erfinv",
    "exp2",
    "expand",
    "exponential",
    "fft.fft2",
    "fft.fft",
    "fft.fftn",
    "fft.hfft2",
    "fft.hfft",
    "fft.hfftn",
    "fft.ifft2",
    "fft.ifft",
    "fft.ifftn",
    "fft.ihfft2",
    "fft.ihfft",
    "fft.ihfftn",
    "fft.irfft2",
    "fft.irfft",
    "fft.irfftn",
    "fft.rfft2",
    "fft.rfft",
    "fft.rfftn",
    "floor_divide",
    "gather",
    "gcd",
    "geometric",
    "geqrf",
    "grid_sampler_2d",
    "heaviside",
    "histc",
    "histogram",
    "histogramdd",
    "hypot",
    "igamma",
    "igammac",
    "index_copy",
    "index_fill",
    "index_put",
    "index_reduce",
    "index_select",
    "isclose",
    "kthvalue",
    "lgamma",
    "linalg.cholesky",
    "linalg.cholesky_ex",
    "linalg.cond",
    "linalg.cross",
    "linalg.det",
    "linalg.eig",
    "linalg.eigh",
    "linalg.eigvals",
    "linalg.eigvalsh",
    "linalg.householder_product",
    "linalg.inv",
    "linalg.inv_ex",
    "linalg.ldl_factor",
    "linalg.ldl_factor_ex",
    "linalg.ldl_solve",
    "linalg.lstsq",
    "linalg.lu",
    "linalg.lu_factor",
    "linalg.lu_factor_ex",
    "linalg.lu_solve",
    "linalg.matrix_norm",
    "linalg.matrix_power",
    "linalg.matrix_rank",
    "linalg.multi_dot",
    "linalg.norm",
    "linalg.pinv",
    "linalg.slogdet",
    "linalg.solve",
    "linalg.solve_ex",
    "linalg.solve_triangular",
    "linalg.svd",
    "linalg.svdvals",
    "linalg.tensorinv",
    "linalg.tensorsolve",
    "linalg.vander",
    "linalg.vector_norm",
    "linspace",
    "log_normal",
    "log_softmax",
    "logaddexp2",
    "logaddexp",
    "logcumsumexp",
    "logdet",
    "logspace",
    "lu",
    "lu_solve",
    "lu_unpack",
    "masked.amax",
    "masked.amin",
    "masked.argmax",
    "masked.argmin",
    "masked.cumsum",
    "masked.log_softmax",
    "masked.logaddexp",
    "masked.logsumexp",
    "masked.mean",
    "masked.median",
    "masked.norm",
    "masked.normalize",
    "masked.prod",
    "masked_scatter",
    "masked_select",
    "masked.softmax",
    "masked.softmin",
    "masked.std",
    "masked.sum",
    "masked.var",
    "matrix_exp",
    "matmul",
    "max_pool2d_with_indices_backward",
    "max",
    "median",
    "min",
    "mode",
    "multinomial",
    "mvlgamma",
    "nanmedian",
    "nanquantile",
    "nansum",
    "narrow_copy",
    "narrow",
    "native_layer_norm",
    "new_empty",
    "new_empty_strided",
    "nextafter",
    "nn.functional.adaptive_avg_pool1d",
    "nn.functional.adaptive_avg_pool2d",
    "nn.functional.adaptive_avg_pool3d",
    "nn.functional.adaptive_max_pool1d",
    "nn.functional.adaptive_max_pool2d",
    "nn.functional.adaptive_max_pool3d",
    "nn.functional.alpha_dropout",
    "nn.functional.avg_pool1d",
    "nn.functional.avg_pool2d",
    "nn.functional.avg_pool3d",
    "nn.functional.batch_norm",
    "nn.functional.bilinear",
    "nn.functional.binary_cross_entropy",
    "nn.functional.conv2d",
    "nn.functional.conv3d",
    "nn.functional.conv_transpose1d",
    "nn.functional.conv_transpose2d",
    "nn.functional.conv_transpose3d",
    "nn.functional.cosine_embedding_loss",
    "nn.functional.cosine_similarity",
    "nn.functional.cross_entropy",
    "nn.functional.ctc_loss",
    "nn.functional.dropout2d",
    "nn.functional.dropout3d",
    "nn.functional.dropout",
    "nn.functional.embedding_bag",
    "nn.functional.embedding",
    "nn.functional.feature_alpha_dropout",
    "nn.functional.fractional_max_pool2d",
    "nn.functional.fractional_max_pool3d",
    "nn.functional.gaussian_nll_loss",
    "nn.functional.grid_sample",
    "nn.functional.group_norm",
    "nn.functional.hinge_embedding_loss",
    "nn.functional.instance_norm",
    "nn.functional.interpolate",
    "nn.functional.layer_norm",
    "nn.functional.leaky_relu",
    "nn.functional.linear",
    "nn.functional.logsigmoid",
    "nn.functional.margin_ranking_loss",
    "nn.functional.max_pool1d",
    "nn.functional.max_pool2d",
    "nn.functional.max_pool3d",
    "nn.functional.max_unpool1d",
    "nn.functional.max_unpool2d",
    "nn.functional.max_unpool3d",
    "nn.functional.multi_head_attention_forward",
    "nn.functional.multi_margin_loss",
    "nn.functional.multilabel_margin_loss",
    "nn.functional.multilabel_soft_margin_loss",
    "nn.functional.nll_loss",
    "nn.functional.normalize",
    "nn.functional.one_hot",
    "nn.functional.pad",
    "nn.functional.pairwise_distance",
    "nn.functional.pixel_shuffle",
    "nn.functional.pixel_unshuffle",
    "nn.functional.poisson_nll_loss",
    "nn.functional.rrelu",
    "nn.functional.scaled_dot_product_attention",
    "nn.functional.softmin",
    "nn.functional.unfold",
    "nn.functional.upsample_nearest",
    "nonzero",
    "nonzero_static",
    "norm",
    "normal",
    "ormqr",
    "pca_lowrank",
    "pinverse",
    "polar",
    "polygamma",
    "prod",
    "put",
    "quantile",
    "renorm",
    "repeat_interleave",
    "resize_",
    "resize_as_",
    "rot90",
    "rsub",
    "scatter_add",
    "scatter",
    "scatter_reduce",
    "searchsorted",
    "select_scatter",
    "signbit",
    "softmax",
    "sort",
    "special.airy_ai",
    "special.i1",
    "special.i1e",
    "special.laguerre_polynomial_l",
    "special.log_ndtr",
    "special.modified_bessel_i0",
    "special.modified_bessel_i1",
    "special.modified_bessel_k0",
    "special.modified_bessel_k1",
    "special.ndtri",
    "special.polygamma",
    "special.scaled_modified_bessel_k0",
    "special.scaled_modified_bessel_k1",
    "special.spherical_bessel_j0",
    "special.zeta",
    "squeeze",
    "stft",
    "sub",
    "svd",
    "svd_lowrank",
    "take_along_dim",
    "take",
    "tensor_split",
    "to_sparse",
    "topk",
    "trace",
    "triangular_solve",
    "triu",
    "unbind",
    "unfold_copy",
    "unfold",
    "unique_consecutive",
    "unique",
    "unravel_index",
    "var_mean",
    "zero_",
    "argwhere",
    "cumulative_trapezoid",
    "expand_as",
    "mean",
    "nanmean",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "bfloat16",
    "bmm",
    "broadcast_shapes",
    "cartesian_prod",
    "cdouble",
    "ceil",
    "chalf", # Skip due to jax not support complex32 with backend: https://github.com/google/jax/issues/14180
    "expm1",
    "fft.fftshift",
    "fft.ifftshift",
    "fill",
    "nn.functional.smooth_l1_loss",
    "nn.functional.soft_margin_loss",
    "nn.functional.softplus",
    "nn.functional.softshrink",
    "nn.functional.softsign",
    "nn.functional.tanhshrink",
    "nn.functional.threshold",
    "nn.functional.triplet_margin_loss",
    "nn.functional.triplet_margin_with_distance_loss",
    "nn.functional.upsample_bilinear",
    "outer",
    "permute",
    "positive",
    "rad2deg",
    "randint",
    "ravel",
    "reciprocal",
    "remainder",
    "repeat",
    "true_divide",
    "trunc",
    "unflatten",
    "unsafe_chunk",
    "unsafe_split",
    "unsqueeze",
    "view_as_complex",
    "view_as",
}

# These inputs are themselves views
# We cannot know how are the views created so cannot replicate the behavior.
variant_test_name_to_skip = {
  "partial_views",
}

random_ops = {
  'empty',
  'empty_like',
  'empty_permuted',
  'empty_strided',
  'bernoulli',
  'randint_like',
  'randn',
  'randn_like',
  'rand',
  'rand_like',
  'uniform',
}

def diff_output(testcase, output1, output2, rtol, atol, equal_nan=True, check_output=True):
  if isinstance(output1, torch.Tensor):
    testcase.assertIsInstance(output2, torch.Tensor)
    output2_cpu = output2.detach().cpu()
    if output1.layout != torch.strided:
      # We only compare dense tensors. We dont currently support sparse tensors
      output1 = output1.to_dense()
    if check_output:
      torch.testing.assert_close(
          output2_cpu, output1, rtol=rtol, atol=atol, equal_nan=equal_nan)
    else:
      testcase.assertEqual(
        (output1.shape, output1.dtype),
        (output2.shape, output2.dtype)
      )
  elif isinstance(output1, (tuple, list)):
    testcase.assertIsInstance(output2, (tuple, list))
    testcase.assertEqual(len(output1), len(output2))
    for o1, o2 in zip(output1, output2):
      diff_output(testcase, o1, o2, rtol, atol)
  else:
    testcase.assertEqual(output1, output2)


def run_export_and_compare(testcase,
                           func,
                           sample_input,
                           check_output=True,
                           equal_nan=True,
                           ignore_indices=False):
  atol = 1e-3
  rtol = 1e-5
  with testcase.subTest("torch_eval"):
    res = func(sample_input.input, *sample_input.args, **sample_input.kwargs)
    with testcase.subTest("torch_xla2_eval"):
      input2, args2, kwargs2 = testcase.env.to_xla((
        sample_input.input, sample_input.args, sample_input.kwargs))
      with testcase.env:
        res2 = func(input2, *args2, **kwargs2)
      res2 = pytree.tree_map_only(tensor.XLATensor2, lambda t: t.torch(), res2)
      with testcase.subTest("torch_xla2_diff:" + str(atol)):
        if ignore_indices and isinstance(res, tuple) and len(res) == 2:
          diff_output(
              testcase,
              res[0],
              res2[0],
              atol=atol,
              rtol=rtol,
              equal_nan=equal_nan, check_output=check_output)
        else:
          diff_output(
              testcase, res, res2, atol=atol, rtol=rtol, equal_nan=equal_nan, check_output=check_output)


ops_to_test = [
    test for test in op_db
    if (test.name not in skiplist and
        test.variant_test_name not in variant_test_name_to_skip)
]


class TestOpInfo(TestCase):

  @classmethod
  def setUpClass(cls):
    print('op_db size: ', len(op_db), 'testing: ', len(ops_to_test))

  def setUp(self):
    self.env = tensor.Environment()

  @ops(ops_to_test, allowed_dtypes=(torch.float32, torch.long))
  def test_reference_eager(self, device, dtype, op):
    sample_inputs = op.sample_inputs(device, dtype)
    for sample_input in sample_inputs:
      t = sample_input.input
      if isinstance(t, torch.Tensor) and t.is_sparse:
        continue
      check_output = op.name not in random_ops
      run_export_and_compare(self, op, sample_input, check_output)


instantiate_device_type_tests(TestOpInfo, globals())

if __name__ == '__main__':
  unittest.main()
