import unittest

import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, ops)
from torch.utils import _pytree as pytree
from torch_xla2 import tensor
import torch_xla2


skiplist = {
    "__rpow__",  # NOTE: cannot fix because torch test case has undefined behavior
                 # such as 0 to negative power.
    "_segment_reduce",
    "_upsample_bilinear2d_aa",
    "bincount", # NOTE: dtype for int input torch gives float. This is weird.
    "byte",
    "cat",
    "cauchy",
    "cdist",
    "ceil",
    "cholesky",
    "cholesky_inverse",
    "cholesky_solve",
    "complex",
    "diagonal_copy",
    "diagonal_scatter",
    "digamma",
    "exponential",
    "gcd",
    "geometric",
    "geqrf",
    "histogram", # hard op: AssertionError: Tensor-likes are not close!
    "histogramdd", # TypeError: histogram requires ndarray or scalar arguments, got <class 'list'> at position 1.
    "igammac",
    "index_reduce",
    "kthvalue",
    "lgamma",
    "linalg.cholesky",
    "linalg.cholesky_ex",
    "linalg.det",
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
    "linalg.solve_ex",
    "linalg.solve_triangular",
    "linalg.svd",
    "linalg.svdvals",
    "linalg.tensorinv",
    "linalg.tensorsolve",
    "linalg.vector_norm",
    "linspace",
    "log_normal",
    "logspace",
    "lu",
    "lu_solve",
    "lu_unpack",
    "masked.median",
    "max_pool2d_with_indices_backward",
    "min",
    "mode",
    "multinomial",
    "mvlgamma",
    "nanmedian",
    "new_empty",
    "new_empty_strided",
    "nextafter",
    "nn.functional.adaptive_avg_pool3d",
    "nn.functional.adaptive_max_pool1d",
    "nn.functional.adaptive_max_pool2d",
    "nn.functional.adaptive_max_pool3d",
    "nn.functional.alpha_dropout",
    "nn.functional.avg_pool1d",
    "nn.functional.avg_pool2d",
    "nn.functional.avg_pool3d",
    "nn.functional.bilinear",
    "nn.functional.conv_transpose1d",
    "nn.functional.conv_transpose2d",
    "nn.functional.conv_transpose3d",
    "nn.functional.cosine_embedding_loss",
    "nn.functional.cosine_similarity",
    "nn.functional.ctc_loss",
    "nn.functional.dropout2d",
    "nn.functional.dropout3d",
    "nn.functional.dropout",
    "nn.functional.embedding_bag",
    "nn.functional.embedding",
    "nn.functional.fractional_max_pool2d",
    "nn.functional.fractional_max_pool3d",
    "nn.functional.group_norm",
    "nn.functional.hinge_embedding_loss",
    "nn.functional.interpolate",
    "nn.functional.margin_ranking_loss",
    "nn.functional.max_pool1d",
    "nn.functional.max_pool2d",
    "nn.functional.max_pool3d",
    "nn.functional.multi_head_attention_forward",
    "nn.functional.multi_margin_loss",
    "nn.functional.multilabel_margin_loss",
    "nn.functional.pad",
    "nn.functional.pairwise_distance",
    "nn.functional.poisson_nll_loss",
    "nn.functional.rrelu",
    "nn.functional.triplet_margin_loss",
    "nn.functional.triplet_margin_with_distance_loss",
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
    "rsub",
    "searchsorted",
    "special.airy_ai",
    "special.scaled_modified_bessel_k0",
    "special.scaled_modified_bessel_k1",
    "special.spherical_bessel_j0",
    "special.zeta",
    "stft",
    "sub",
    "svd",
    "svd_lowrank",
    "to_sparse", # We are not supporting sparse tensors yet.
    "unfold_copy",
    "unfold",
    "unique_consecutive",
    "unique",
    "unravel_index",
    "trunc",
    "var_mean",
    "argwhere",
    "nanmean",
    "chalf", # Skip due to jax not support complex32 with backend: https://github.com/google/jax/issues/14180
    "nn.functional.upsample_bilinear",
    "randint",
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
  # Dropout is not deterministic https://pytorch.org/docs/stable/generated/torch.nn.functional.feature_alpha_dropout.html
  'nn.functional.feature_alpha_dropout',
}

atol_dict = {"matrix_exp": (2e-1, 2e-4), "linalg.pinv": (8e-1, 2e0), "linalg.eig": (2e0, 3e0), "linalg.eigh": (5e1, 3e0), "linalg.eigvalsh": (5e1, 3e0)}

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
  atol, rtol = (1e-3, 1e-5)
  if func.name in atol_dict:
    atol, rtol = atol_dict[func.name]
  
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

# Sort related ops should ignore index;
# For example: sort( [1, 0, 0]) -> [0, 0, 1]
# the correct index can be [1, 2, 0] or [2, 1, 0]
should_ignore_indexes = {
  "topk"
}


class TestOpInfo(TestCase):

  @classmethod
  def setUpClass(cls):
    print('op_db size: ', len(op_db), 'testing: ', len(ops_to_test))

  def setUp(self):
    self.env = torch_xla2.default_env()
    torch_xla2.enable_accuracy_mode()
    #self.env.config.debug_accuracy_for_each_op = True 
    torch.manual_seed(0)

  # Replaces all values in the input torch_tensor that are less than the given threshold
  # with the threshold value itself.
  def replace_values_below_threshold(self, torch_tensor, threshold):
      return torch.where(torch_tensor < threshold, torch.tensor(threshold), torch_tensor)

  @ops(ops_to_test, allowed_dtypes=(torch.float32, torch.long))
  def test_reference_eager(self, device, dtype, op):
    sample_inputs = op.sample_inputs(device, dtype)
    for sample_input in sample_inputs:
      t = sample_input.input
      if isinstance(t, torch.Tensor) and t.is_sparse:
        continue
      check_output = op.name not in random_ops

      if op.name == "special.polygamma":
        # The polygamma function is inaccurate for values < 1.
        # To avoid errors during testing, replace values below 1 with 1.
        sample_input.input = self.replace_values_below_threshold(
            sample_input.input, 1)
      if op.name == "nn.functional.scaled_dot_product_attention":
        check_output = sample_input.kwargs.get('dropout_p') == 0.0
      
      ignore_index = op.name in should_ignore_indexes

      run_export_and_compare(self, op, sample_input, check_output, 
                             ignore_indices=ignore_index)


instantiate_device_type_tests(TestOpInfo, globals())

if __name__ == '__main__':
  unittest.main()
