import logging
import unittest

from typing import Optional, Union, Callable

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from torch_xla.experimental.custom_kernel import gmm, tgmm, gmm_backward, GMM
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

  def _reference_gmm(self, lhs: torch.Tensor, rhs: torch.Tensor,
                     group_sizes: torch.Tensor) -> torch.Tensor:
    start = 0
    out = []
    for i, size in enumerate(group_sizes):
      result = lhs[start:start + size, :] @ rhs[i, :, :]
      out.append(result)
      start += group_sizes[i]
    return torch.cat(out)

  def _reference_tgmm(self, lhs: torch.Tensor, rhs: torch.Tensor,
                      group_sizes: torch.Tensor) -> torch.Tensor:
    start = 0
    out = []
    for i, size in enumerate(group_sizes):
      result = lhs[:, start:start + size] @ rhs[start:start + size, :]
      out.append(result)
      start += group_sizes[i]
    return torch.stack(out)

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
    jax.config.update('jax_default_matmul_precision', "highest")
    compiled_gmm = torch.compile(torch.ops.xla.gmm, backend="openxla")
    gmm_funcs = [
        gmm,
        torch.ops.xla.gmm,
        compiled_gmm,
    ]

    self._init_test_cases()
    for test_cache in [False, True]:
      for gmm_func in gmm_funcs:
        for test_case in self.tests_cases:
          num_groups = test_case['num_groups']
          k = test_case['k']
          m = test_case['m']
          n = test_case['n']
          lhs_dtype = rhs_dtype = test_case['dtype']

          lhs = torch.rand(m, k, dtype=lhs_dtype)
          rhs = torch.rand(num_groups, k, n, dtype=rhs_dtype)
          group_sizes = self._group_sizes_strategy(m=m, num_groups=num_groups)
          ref_out = self._reference_gmm(lhs, rhs, group_sizes)

          out = gmm_func(lhs.to("xla"), rhs.to("xla"), group_sizes.to("xla"))
          # torch.compiled version of the gmm will cache the payload in dynamo layer
          # hence won't trigger the trace_pallas cache
          if test_cache and gmm_func != compiled_gmm:
            old_cnt = xr.get_num_cached_compilation_graph()
            # execute the same gmm func, expected to hit the cache
            out = gmm_func(lhs.to("xla"), rhs.to("xla"), group_sizes.to("xla"))
            new_cnt = xr.get_num_cached_compilation_graph()
            self.assertEqual(old_cnt, new_cnt)
          self.assertTrue(torch.allclose(ref_out, out.cpu()))

    # Make sure gmm doesn't fallback.
    self.assertEqual(len(torch_xla._XLAC._get_executed_fallback_ops()), 0)
    jax.config.update('jax_default_matmul_precision', "default")

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_gmm_bf16(self):
    met.clear_all()

    compiled_gmm = torch.compile(torch.ops.xla.gmm, backend="openxla")
    gmm_funcs = [gmm, torch.ops.xla.gmm, compiled_gmm]
    self._init_test_cases()
    for test_cache in [False, True]:
      for gmm_func in gmm_funcs:
        for test_case in self.tests_cases:
          num_groups = test_case['num_groups']
          k = test_case['k']
          m = test_case['m']
          n = test_case['n']
          lhs_dtype = rhs_dtype = torch.bfloat16

          lhs = torch.rand(m, k, dtype=lhs_dtype)
          rhs = torch.rand(num_groups, k, n, dtype=rhs_dtype)
          group_sizes = self._group_sizes_strategy(m=m, num_groups=num_groups)
          ref_out = self._reference_gmm(lhs, rhs, group_sizes)

          out = gmm_func(lhs.to("xla"), rhs.to("xla"), group_sizes.to("xla"))
          # torch.compiled version of the gmm will cache the payload in dynamo layer
          # hence won't trigger the trace_pallas cache
          if test_cache and gmm_func != compiled_gmm:
            old_cnt = xr.get_num_cached_compilation_graph()
            # execute the same gmm func, expected to hit the cache
            out = gmm_func(lhs.to("xla"), rhs.to("xla"), group_sizes.to("xla"))
            new_cnt = xr.get_num_cached_compilation_graph()
            self.assertEqual(old_cnt, new_cnt)
          self.assertTrue(torch.allclose(ref_out, out.cpu()))

    # Make sure gmm doesn't fallback.
    self.assertEqual(len(torch_xla._XLAC._get_executed_fallback_ops()), 0)

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_tgmm(self):
    met.clear_all()
    jax.config.update('jax_default_matmul_precision', "highest")

    self._init_test_cases()
    for test_cache in [False, True]:
      for test_case in self.tests_cases:
        num_groups = test_case['num_groups']
        k = test_case['k']
        m = test_case['m']
        n = test_case['n']
        lhs_dtype = rhs_dtype = test_case['dtype']

        lhs = torch.rand(k, m, dtype=lhs_dtype)
        rhs = torch.rand(m, n, dtype=rhs_dtype)
        group_sizes = self._group_sizes_strategy(m=m, num_groups=num_groups)
        ref_out = self._reference_tgmm(lhs, rhs, group_sizes)

        out = tgmm(lhs.to("xla"), rhs.to("xla"), group_sizes.to("xla"))
        if test_cache:
          old_cnt = xr.get_num_cached_compilation_graph()
          # execute the same gmm func, expected to hit the cache
          out = tgmm(lhs.to("xla"), rhs.to("xla"), group_sizes.to("xla"))
          new_cnt = xr.get_num_cached_compilation_graph()
          self.assertEqual(new_cnt, old_cnt)
        self.assertTrue(torch.allclose(ref_out, out.cpu()))

    # Make sure tgmm doesn't fallback.
    self.assertEqual(len(torch_xla._XLAC._get_executed_fallback_ops()), 0)
    jax.config.update('jax_default_matmul_precision', "default")

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_tgmm_bf16(self):
    met.clear_all()

    self._init_test_cases()
    for test_cache in [False, True]:
      for test_case in self.tests_cases:
        num_groups = test_case['num_groups']
        k = test_case['k']
        m = test_case['m']
        n = test_case['n']
        lhs_dtype = rhs_dtype = torch.bfloat16

        lhs = torch.rand(k, m, dtype=lhs_dtype)
        rhs = torch.rand(m, n, dtype=rhs_dtype)
        group_sizes = self._group_sizes_strategy(m=m, num_groups=num_groups)
        ref_out = self._reference_tgmm(lhs, rhs, group_sizes)

        out = tgmm(lhs.to("xla"), rhs.to("xla"), group_sizes.to("xla"))
        if test_cache:
          old_cnt = xr.get_num_cached_compilation_graph()
          # execute the same gmm func, expected to hit the cache
          out = tgmm(lhs.to("xla"), rhs.to("xla"), group_sizes.to("xla"))
          new_cnt = xr.get_num_cached_compilation_graph()
          self.assertEqual(new_cnt, old_cnt)
        self.assertTrue(torch.allclose(ref_out, out.cpu()))

    # Make sure tgmm doesn't fallback.
    self.assertEqual(len(torch_xla._XLAC._get_executed_fallback_ops()), 0)

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_gmm_backward(self):
    self._init_test_cases()
    for test_case in self.tests_cases:
      num_groups = test_case['num_groups']
      k = test_case['k']
      m = test_case['m']
      n = test_case['n']
      lhs_dtype = rhs_dtype = torch.bfloat16

      for test_cache in [False, True]:
        old_cnt = xr.get_num_cached_compilation_graph()
        lhs = torch.rand(m, k, dtype=lhs_dtype, requires_grad=True)
        rhs = torch.rand(num_groups, k, n, dtype=rhs_dtype, requires_grad=True)
        group_sizes = self._group_sizes_strategy(m=m, num_groups=num_groups)
        lhs.retain_grad()
        rhs.retain_grad()

        ref_out = self._reference_gmm(lhs, rhs, group_sizes)
        ref_out.sum().backward()

        ref_out_backward = torch.ones_like(ref_out)
        grad_lhs, grad_rhs = gmm_backward(
            ref_out_backward.to("xla"), lhs.to("xla"), rhs.to("xla"),
            group_sizes.to("xla"))
        # same gmm/tgmm was run for the `test_cache=False` case so the
        # cache should be populated now
        new_cnt = xr.get_num_cached_compilation_graph()
        if test_cache:
          self.assertEqual(new_cnt, old_cnt)

        self.assertTrue(torch.allclose(lhs.grad, grad_lhs.cpu()))
        self.assertTrue(torch.allclose(rhs.grad, grad_rhs.cpu()))

    # Make sure gmm doesn't fallback.
    self.assertEqual(len(torch_xla._XLAC._get_executed_fallback_ops()), 0)

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_gmm_backward_2(self):
    self._init_test_cases()
    for test_case in self.tests_cases:
      num_groups = test_case['num_groups']
      k = test_case['k']
      m = test_case['m']
      n = test_case['n']
      lhs_dtype = rhs_dtype = torch.bfloat16

      torch.manual_seed(42)
      lhs = torch.rand(m, k, dtype=lhs_dtype, requires_grad=True)
      rhs = torch.rand(num_groups, k, n, dtype=rhs_dtype, requires_grad=True)
      group_sizes = self._group_sizes_strategy(m=m, num_groups=num_groups)
      lhs.retain_grad()
      rhs.retain_grad()

      ref_out = self._reference_gmm(lhs, rhs, group_sizes)
      ref_out.sum().backward()

      torch.manual_seed(42)
      lhs_xla = torch.rand(m, k, dtype=lhs_dtype, requires_grad=True).to("xla")
      rhs_xla = torch.rand(
          num_groups, k, n, dtype=rhs_dtype, requires_grad=True).to("xla")
      lhs_xla.retain_grad()
      rhs_xla.retain_grad()

      out = GMM.apply(lhs_xla, rhs_xla, group_sizes.to("xla"))
      out.sum().backward()

      self.assertTrue(torch.allclose(ref_out, out.cpu()))
      self.assertTrue(torch.allclose(lhs.grad, lhs_xla.grad.cpu()))
      self.assertTrue(torch.allclose(rhs.grad, rhs_xla.grad.cpu()))

    # Make sure gmm doesn't fallback.
    self.assertEqual(len(torch_xla._XLAC._get_executed_fallback_ops()), 0)

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_gmm_backward_3(self):
    self._init_test_cases()
    for test_case in self.tests_cases:
      num_groups = test_case['num_groups']
      k = test_case['k']
      m = test_case['m']
      n = test_case['n']
      lhs_dtype = rhs_dtype = torch.bfloat16

      torch.manual_seed(42)
      lhs = torch.rand(m, k, dtype=lhs_dtype, requires_grad=True)
      rhs = torch.rand(num_groups, k, n, dtype=rhs_dtype, requires_grad=True)
      group_sizes = self._group_sizes_strategy(m=m, num_groups=num_groups)
      lhs.retain_grad()
      rhs.retain_grad()

      ref_out = self._reference_gmm(lhs, rhs, group_sizes)
      ref_out.sum().backward()

      torch.manual_seed(42)
      lhs_xla = torch.rand(m, k, dtype=lhs_dtype, requires_grad=True).to("xla")
      rhs_xla = torch.rand(
          num_groups, k, n, dtype=rhs_dtype, requires_grad=True).to("xla")
      lhs_xla.retain_grad()
      rhs_xla.retain_grad()

      out = GMM.apply(lhs_xla, rhs_xla, group_sizes.to("xla"))
      grad_out = torch.ones_like(out)
      torch.autograd.backward([out], [grad_out, lhs_xla, rhs_xla])

      self.assertTrue(torch.allclose(ref_out, out.cpu()))
      self.assertTrue(torch.allclose(lhs.grad, lhs_xla.grad.cpu()))
      self.assertTrue(torch.allclose(rhs.grad, rhs_xla.grad.cpu()))

    # Make sure gmm doesn't fallback.
    self.assertEqual(len(torch_xla._XLAC._get_executed_fallback_ops()), 0)

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_gmm_cache_miss(self):
    met.clear_all()
    jax.config.update('jax_default_matmul_precision', "highest")

    self._init_test_cases()
    test_case = self.tests_cases[-1]
    # make sure that cache miss for different input shapes and dtype
    met.clear_all()
    for mul_factor in [[2, 1, 1, 1], [1, 2, 1, 1], [2, 1, 2, 1], [2, 1, 1, 2]]:
      for dtype in [torch.float32, torch.bfloat16]:
        for tiling in [(128, 128, 128), (256, 256, 256)]:
          num_groups = test_case['num_groups'] * mul_factor[0]
          k = test_case['k'] * mul_factor[1]
          m = test_case['m'] * mul_factor[2]
          n = test_case['n'] * mul_factor[3]
          lhs_dtype = rhs_dtype = dtype

          lhs = torch.rand(m, k, dtype=lhs_dtype)
          rhs = torch.rand(num_groups, k, n, dtype=rhs_dtype)
          group_sizes = self._group_sizes_strategy(m=m, num_groups=num_groups)

          out = gmm(lhs.to("xla"), rhs.to("xla"), group_sizes.to("xla"), tiling)
          self.assertEqual(met.counter_value('trace_pallas_cache_hit'), None)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  torch.set_default_dtype(torch.float32)
  torch.manual_seed(42)
  torch_xla._XLAC._xla_set_mat_mul_precision('highest')
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
