import functools
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import tempfile

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from absl.testing import absltest, parameterized
from torch import nn


class M(nn.Module):

  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(5, 3)

  def forward(self, x):
    return self.linear(x)


def _test_spawn(fn, args):
  # Use a new process for each test to clear compilation cache state.
  with ProcessPoolExecutor() as pool:
    pool.submit(fn, *args)


class TestGraphHash(parameterized.TestCase):

  def _test_num_graph_hash(self, use_dynamo, use_persistent):
    xla_dev = torch_xla.device()
    model = M().to(device=xla_dev)
    input_shape = (10, 5)
    if use_dynamo:
      model = torch.compile(model, backend='openxla')
    if use_persistent:
      # Init persistent cache.
      tmpdir = tempfile.TemporaryDirectory()
      xr.initialize_cache(tmpdir)
    input1 = torch.rand(input_shape).to(xla_dev)
    torch_xla.sync()
    model(input1)
    torch_xla.sync()
    xm.wait_device_ops()
    graph_cnt = xr.get_num_cached_compilation_graph()
    input2 = torch.rand(input_shape).to(xla_dev)
    model(input2)
    torch_xla.sync()
    xm.wait_device_ops()
    new_graph_cnt = xr.get_num_cached_compilation_graph()
    # No compilation happening since same graph runs.
    self.assertEqual(graph_cnt, new_graph_cnt)
    graph_cnt = new_graph_cnt
    input3 = torch.concat([input1, input2], dim=0)
    torch_xla.sync()
    xm.wait_device_ops()
    new_graph_cnt = xr.get_num_cached_compilation_graph()
    # Stacking the inputs creates a new graph.
    self.assertEqual(graph_cnt + 1, new_graph_cnt)
    graph_cnt = new_graph_cnt
    model(input3)
    torch_xla.sync()
    xm.wait_device_ops()
    new_graph_cnt = xr.get_num_cached_compilation_graph()
    # New compilation with stacked inputs.
    self.assertEqual(graph_cnt + 1, new_graph_cnt)

  @parameterized.product(
      use_dynamo=(True, False),
      use_persistent=(True, False),
  )
  def test_num_graph_hash(self, use_dynamo, use_persistent):
    if use_persistent and (xr.device_type() not in {'TPU', 'NEURON'}):
      raise absltest.SkipTest('Device type does not support persistent caching')
    _test_spawn(self._test_num_graph_hash, (use_dynamo, use_persistent))


class TestClearCompilationCache(parameterized.TestCase):

  def _test_clear_memory_cache(self):
    """Test clearing the in-memory compilation cache."""
    xla_dev = torch_xla.device()
    model = M().to(device=xla_dev)
    input1 = torch.rand((10, 5)).to(xla_dev)
    model(input1)
    torch_xla.sync()
    xm.wait_device_ops()

    graph_cnt = xr.get_num_cached_compilation_graph()
    self.assertGreater(graph_cnt, 0)

    xr.clear_compilation_cache()

    new_graph_cnt = xr.get_num_cached_compilation_graph()
    self.assertEqual(new_graph_cnt, 0)

  def test_clear_memory_cache(self):
    _test_spawn(self._test_clear_memory_cache, ())

  def _test_clear_persistent_cache(self):
    """Test clearing the persistent compilation cache (memory + disk)."""
    with tempfile.TemporaryDirectory() as tmpdir:
      xr.initialize_cache(tmpdir)

      xla_dev = torch_xla.device()
      model = M().to(device=xla_dev)
      input1 = torch.rand((10, 5)).to(xla_dev)
      model(input1)
      torch_xla.sync()
      xm.wait_device_ops()

      graph_cnt = xr.get_num_cached_compilation_graph()
      self.assertGreater(graph_cnt, 0)

      cache_files_before = os.listdir(tmpdir)
      self.assertGreater(len(cache_files_before), 0)

      xr.clear_compilation_cache()

      new_graph_cnt = xr.get_num_cached_compilation_graph()
      self.assertEqual(new_graph_cnt, 0)

      cache_files_after = os.listdir(tmpdir)
      self.assertEqual(len(cache_files_after), 0)

  def test_clear_persistent_cache(self):
    if xr.device_type() not in {'TPU', 'NEURON'}:
      raise absltest.SkipTest('Device type does not support persistent caching')
    _test_spawn(self._test_clear_persistent_cache, ())


if __name__ == '__main__':
  test = absltest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
