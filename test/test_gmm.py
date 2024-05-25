import logging
import unittest

from typing import Optional, Union, Callable

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from torch_xla.experimental.custom_kernel import gmm, _make_group_metadata, _histogram
from torch_xla import runtime as xr
from torch_xla._internal import tpu

import numpy as np

if xr.device_type() == 'TPU':
  from torch_xla.experimental.custom_kernel import jax_import_guard
  jax_import_guard()
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl


class MegabloxTest(unittest.TestCase):

  def _reference_gmm(
      self,
      lhs: np.array,
      rhs: np.array,
      group_sizes: np.array,
      preferred_element_type: np.dtype = np.float32,
  ) -> np.array:

    start = 0
    out = []
    for i, size in enumerate(group_sizes):
      result = np.dot(lhs[start:start + size, :], rhs[i, :, :])

      result = result.astype(preferred_element_type)
      out.append(result)
      start += group_sizes[i]
    return np.array(np.concatenate(out, axis=0))

  def _group_sizes_strategy(self, m: int, num_groups: int) -> torch.Tensor:
    # Randomly sample the ends of the groups in the m-dimension. Let the fuzzer
    # sample with replacement so that it's possible to get zero-sized groups. Get
    # 'num_groups - 1' run ends. The final group will end at 'm'.
    ends_no_final = np.sort(
        np.array(
            [np.random.randint(low=0, high=m) for _ in range(num_groups - 1)],
            dtype=np.int32,
        ),)
    ends = np.concatenate([ends_no_final, np.array([m], dtype=np.int32)])

    # Calculate the run starts by shifting ends 1 to the right. The first run
    # starts at zero.
    starts = np.concatenate([np.zeros(1, dtype=np.int32), ends_no_final])
    return torch.from_numpy(ends - starts).to(torch.int32)

  def _tolerances(self, lhs_dtype: torch.dtype, rhs_dtype: torch.dtype,
                  out_dtype: torch.dtype) -> tuple[float, float]:
    if (lhs_dtype == torch.bfloat16 or rhs_dtype == torch.bfloat16 or
        out_dtype == torch.bfloat16):
      return 1e-3, 1e-2  # atol, rtol
    return 1e-4, 1e-2  # atol, rtol

  LutFn = Callable[[int, int, int], Optional[tuple[int, int, int]]]

  def _init_test_cases(self):
    self.tests_cases = []
    self.tests_cases.append({
        'dtype': torch.float32,
        'm': 128,
        'k': 128,
        'n': 128,
        'num_groups': 1
    })
    self.tests_cases.append({
        'dtype': torch.float32,
        'm': 256,
        'k': 128,
        'n': 128,
        'num_groups': 1
    })
    self.tests_cases.append({
        'dtype': torch.float32,
        'm': 128,
        'k': 256,
        'n': 128,
        'num_groups': 8
    })
    self.tests_cases.append({
        'dtype': torch.float32,
        'm': 512,
        'k': 128,
        'n': 256,
        'num_groups': 2
    })

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_gmm(self):
    met.clear_all()

    self._init_test_cases()
    for test_case in self.tests_cases:
      num_groups = test_case['num_groups']
      k = test_case['k']
      m = test_case['m']
      n = test_case['n']
      lhs_dtype = rhs_dtype = test_case['dtype']
      out_dtype = torch.float32

      lhs = torch.rand(m, k, dtype=lhs_dtype).to('xla')
      rhs = torch.rand(num_groups, k, n, dtype=rhs_dtype).to('xla')
      group_sizes = self._group_sizes_strategy(
          m=m, num_groups=num_groups).to('xla')
      out = gmm(lhs, rhs, group_sizes)

      ref_out = self._reference_gmm(lhs.cpu().float().numpy(),
                                    rhs.cpu().float().numpy(),
                                    group_sizes.cpu().numpy())

      atol, rtol = self._tolerances(lhs_dtype, rhs_dtype, out_dtype)
      np.testing.assert_allclose(
          ref_out, np.array(out[0].cpu()), rtol=rtol, atol=atol)

    # Make sure _make_group_metadata doesn't fallback.
    self.assertNotIn("aten::", met.short_metrics_report())

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_make_group_metadata(self):
    from jax.experimental.pallas.ops.tpu.megablox.gmm import make_group_metadata as jax_make_group_metadata
    met.clear_all()

    test_grids = [
        {
            'group_sizes': [8, 8, 8, 8],
            'm': 32,
            'tm': 8
        },
        {
            'group_sizes': [2, 14, 8, 8],
            'm': 32,
            'tm': 8
        },
        {
            'group_sizes': [16, 0, 8, 8],
            'm': 32,
            'tm': 8
        },
        {
            'group_sizes': [2, 0, 14, 16],
            'm': 32,
            'tm': 8
        },
        {
            'group_sizes': [8, 12, 0, 12],
            'm': 32,
            'tm': 8
        },
        {
            'group_sizes': [6, 12, 0, 14],
            'm': 32,
            'tm': 8
        },
        {
            'group_sizes': [6, 12, 0, 14],
            'm': 32,
            'tm': 4
        },
    ]

    for test_grid in test_grids:
      jax_meta, jax_num_tiles = jax_make_group_metadata(
          group_sizes=jnp.array(test_grid['group_sizes']),
          m=test_grid['m'],
          tm=test_grid['tm'],
          start_group=0,
          num_nonzero_groups=len(test_grid['group_sizes']),
      )

      torch_meta = _make_group_metadata(
          group_sizes=torch.tensor(test_grid['group_sizes']).to("xla"),
          m=test_grid['m'],
          tm=test_grid['tm'],
      )

      for i in range(len(jax_meta)):
        self.assertTrue(
            torch.all(torch.from_numpy(np.array(jax_meta[i])) == torch_meta[i].cpu()))
      self.assertEqual(jax_num_tiles, torch_meta[-1].cpu().item())

    # Make sure _make_group_metadata doesn't fallback.
    self.assertNotIn("aten::", met.short_metrics_report())

  def test_histogram(self):
    test_grids = [
        {
            'input': [1, 4, 4, 1, 2, 3],
            'min': 1,
            'max': 4,
        },
        {
            'input': [1, 4, 4, 1, 2, 3],
            'min': 2,
            'max': 3,
        },
        {
            'input': [1, 4, 4, 1, 2, 3],
            'min': 0,
            'max': 5,
        },
        {
            'input': [],
            'min': 0,
            'max': 5,
        },
        {
            'input': [1, 4, 4, 1, 2, 3],
            'min': 2,
            'max': 2,
        },
    ]

    for test_grid in test_grids:
      torch_chart = torch.histc(
          torch.tensor(test_grid['input'], dtype=torch.float),
          bins=test_grid['max'] - test_grid['min'] + 1,
          min=test_grid['min'],
          max=test_grid['max'],
      )

      chart = _histogram(
          torch.tensor(test_grid['input'], dtype=torch.int32).to("xla"),
          min=test_grid['min'],
          max=test_grid['max'],
      )

    self.assertTrue(torch.all(torch_chart == chart.cpu()))

  def test_histogram_raise(self):
    with self.assertRaisesRegex(AssertionError,
                                "input must be of torch.int32 dtype."):
      _histogram(
          torch.tensor([1, 4, 4, 1, 2, 3], dtype=torch.float),
          min=4,
          max=5,
      )

    with self.assertRaisesRegex(AssertionError, "min must be less than max."):
      _histogram(
          torch.tensor([1, 4, 4, 1, 2, 3], dtype=torch.int32),
          min=4,
          max=3,
      )


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  torch.set_default_dtype(torch.float32)
  torch.manual_seed(42)
  torch_xla._XLAC._xla_set_use_full_mat_mul_precision(
      use_full_mat_mul_precision=True)
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
