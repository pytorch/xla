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
    "_segment_reduce",
    "bincount", # NOTE: dtype for int input torch gives float. This is weird.
    "byte",
    "cat",
    "cholesky",
    "cholesky_solve",
    "diagonal_copy",
    "geqrf",
    "histogram", # hard op: AssertionError: Tensor-likes are not close!
    "histogramdd", # TypeError: histogram requires ndarray or scalar arguments, got <class 'list'> at position 1.
    "index_reduce",
    "kthvalue",
    "linalg.cholesky",
    "linalg.cholesky_ex",
    "linalg.det",
    "linalg.ldl_solve",
    "linalg.lu_solve",
    "max_pool2d_with_indices_backward",
    "nn.functional.adaptive_avg_pool3d",
    "nn.functional.adaptive_max_pool1d",
    "nn.functional.adaptive_max_pool2d",
    "nn.functional.adaptive_max_pool3d",
    "nn.functional.alpha_dropout",
    "nn.functional.conv_transpose1d",
    "nn.functional.conv_transpose2d",
    "nn.functional.conv_transpose3d",
    "nn.functional.ctc_loss",
    "nn.functional.dropout2d",
    "nn.functional.dropout3d",
    "nn.functional.dropout",
    "nn.functional.embedding_bag",
    "nn.functional.fractional_max_pool2d",
    "nn.functional.fractional_max_pool3d",
    "nn.functional.interpolate",
    "nn.functional.max_pool1d",
    "nn.functional.max_pool2d",
    "nn.functional.max_pool3d",
    "nn.functional.multi_head_attention_forward",
    "nn.functional.multilabel_margin_loss",
    "nn.functional.pairwise_distance",
    "nn.functional.poisson_nll_loss",
    "nn.functional.rrelu",
    "nn.functional.upsample_nearest",
    "nonzero",
    "nonzero_static",
    "normal",
    "ormqr",
    "pca_lowrank",
    "searchsorted",
    "special.airy_ai",
    "special.scaled_modified_bessel_k0",
    "special.scaled_modified_bessel_k1",
    "special.spherical_bessel_j0",
    "special.zeta",
    "unfold_copy",
    "unfold",
    "randint",
}

not_support_ops_list = {
  "chalf", # Skip due to jax not support complex32 with backend: https://github.com/google/jax/issues/14180
  "__rpow__",  # NOTE: cannot fix because torch test case has undefined behavior
               # such as 0 to negative power.
  "ceil", # only failed with python 3.9
  "trunc", # only failed with python 3.9
  "to_sparse", # We are not supporting sparse tensors yet.
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
  'geometric',
  'new_empty',
  'new_empty_strided',
  'randint_like',
  'randn',
  'randn_like',
  'rand',
  'rand_like',
  'uniform',
  'multinomial',
  # Dropout is not deterministic https://pytorch.org/docs/stable/generated/torch.nn.functional.feature_alpha_dropout.html
  'nn.functional.feature_alpha_dropout',
  'cauchy',
  'exponential',
  'log_normal',
}

atol_dict = {"linalg.eig": (2e0, 3e0),
             "linalg.eigh": (5e1, 3e0),
             "linalg.eigvalsh": (5e1, 3e0),
             "linalg.pinv": (8e-1, 2e0),
             "linalg.svd": (1e0, 1e0),
             "svd": (1e0, 1e0),
             "svd_lowrank": (1e0, 1e0),
             "matrix_exp": (2e-1, 2e-4),
             "cdist": (5e1, 3e0)}

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
    if (test.name not in (skiplist | not_support_ops_list) and
        test.variant_test_name not in variant_test_name_to_skip)
]

# Sort related ops should ignore index;
# For example: sort( [1, 0, 0]) -> [0, 0, 1]
# the correct index can be [1, 2, 0] or [2, 1, 0]
should_ignore_indexes = {
  "topk",
  "mode"
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

      #print("[DEBUG] sample_input: ", sample_input)

      # TODO: this is a workaround to skip int64 cast for linspace
      # reference: https://github.com/pytorch/xla/issues/7505#issuecomment-2400895692 and subsequent comments
      # we have opened a bug in pytorch: https://github.com/pytorch/pytorch/issues/137546
      if op.name == "linspace":
        if 'dtype' in sample_input.kwargs:
          if sample_input.kwargs['dtype'] == torch.int64:
            sample_input.kwargs['dtype'] = torch.float
      if op.name == "polygamma" or op.name == "special.polygamma":
        # The polygamma function is inaccurate for values < 1.
        # To avoid errors during testing, replace values below 1 with 1.
        sample_input.input = self.replace_values_below_threshold(
            sample_input.input, 1)
      if op.name == "nn.functional.scaled_dot_product_attention":
        check_output = sample_input.kwargs.get('dropout_p') == 0.0
      
      ignore_index = op.name in should_ignore_indexes

      run_export_and_compare(self, op, sample_input, check_output, 
                             ignore_indices=ignore_index)


instantiate_device_type_tests(TestOpInfo, globals(), only_for='cpu')

if __name__ == '__main__':
  unittest.main()
