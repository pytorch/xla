import concurrent.futures
import itertools
import os
import queue
import requests
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla import runtime as xr
from torch_xla._internal import pjrt
from absl.testing import absltest, parameterized


@unittest.skipIf(xr.device_type() not in ('GPU', 'CUDA', 'ROCM'),
                 f"GPU tests should only run on GPU devices.")
class TestExperimentalPjrtGpu(parameterized.TestCase):

  def setUp(self):
    xr.set_device_type('CUDA')

    os.environ.update({
        xenv.PJRT_GPU_ASYNC_CLIENT: 'true',
    })

  def test_default_gpu_device(self):
    os.environ.pop(xenv.PJRT_GPU_ASYNC_CLIENT, None)

    num_devices = int(os.environ[xenv.GPU_NUM_DEVICES])
    expected = {i: torch.device(f'xla:0') for i in range(num_devices)}
    devices_per_process = pjrt.run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices_per_process, expected)

  def test_multi_gpu_devices(self):
    num_devices = int(os.environ[xenv.GPU_NUM_DEVICES])
    expected = {i: torch.device(f'xla:0') for i in range(num_devices)}

    devices_per_process = pjrt.run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices_per_process, expected)

  @parameterized.named_parameters(('xla_model', xm.get_ordinal),
                                  ('pjrt', xr.global_ordinal))
  def test_global_ordinal(self, ordinal_func):
    num_devices = int(os.environ[xenv.GPU_NUM_DEVICES])
    expected = [i for i in range(num_devices)]

    results = pjrt.run_multiprocess(ordinal_func)
    self.assertListEqual(sorted(results.values()), expected)

  @parameterized.named_parameters(('xla_model', xm.get_local_ordinal),
                                  ('pjrt', xr.local_ordinal))
  def test_local_ordinal(self, ordinal_func):
    # TODO(wcromar): add multiprocess tests
    num_devices = int(os.environ[xenv.GPU_NUM_DEVICES])
    expected = [i for i in range(num_devices)]

    results = pjrt.run_multiprocess(ordinal_func)
    self.assertListEqual(sorted(results.values()), expected)

  @staticmethod
  def _multi_gpu_backwards():
    results = {}

    class _CustomBackwards(torch.autograd.Function):

      @staticmethod
      def forward(ctx, x):
        ordinal = xm.get_ordinal()
        ctx.forward_ordinal = ordinal
        return x

      @staticmethod
      def backward(ctx, grad_output):
        results['forward_ordinal'] = ctx.forward_ordinal
        results['backward_ordinal'] = xm.get_ordinal()
        results['device'] = str(xm.xla_device())
        return grad_output

    x = torch.ones(1, requires_grad=True, device=xm.xla_device())
    y = _CustomBackwards.apply(x)
    y.backward()
    xm.mark_step()

    return results

  def test_multi_gpu_backwards(self):
    os.environ.update({
        xenv.PJRT_GPU_ASYNC_CLIENT: 'true',
    })
    num_devices = int(os.environ[xenv.GPU_NUM_DEVICES])

    expected = {
        i: {
            'forward_ordinal': i,
            'backward_ordinal': i,
            'device': f'xla:0'
        } for i in range(num_devices)
    }
    results = pjrt.run_multiprocess(self._multi_gpu_backwards)

    self.assertDictEqual(results, expected)

  @staticmethod
  def _spawn(index: int, queue: queue.Queue):
    queue.put(index)

  @parameterized.named_parameters(('xmp', xmp.spawn), ('pjrt', pjrt.spawn))
  def test_spawn(self, spawn):
    manager = torch.multiprocessing.Manager()
    num_devices = int(os.environ[xenv.GPU_NUM_DEVICES])
    queue = manager.Queue(num_devices)
    spawn(self._spawn, args=(queue,))

    indices = sorted(queue.get(block=False) for _ in range(queue.qsize()))
    self.assertListEqual(indices, list(range(num_devices)))

  @staticmethod
  def _broadcast(sync):
    torch.manual_seed(xm.get_ordinal())
    device = xm.xla_device()
    model = nn.Linear(5, 5).to(device)
    if sync:
      xm.broadcast_master_param(model)

    xm.mark_step()
    return next(model.parameters()).detach().cpu().numpy()

  @parameterized.named_parameters(('synchronized_parameters', True),
                                  ('unsynchronized_parameters', False))
  def test_broadcast_master_param(self, sync):
    results = pjrt.run_multiprocess(self._broadcast, sync)
    master_params = results[0]
    for ordinal, worker_params in results.items():
      if sync:
        np.testing.assert_array_equal(master_params, worker_params)
      elif ordinal != 0:
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 master_params, worker_params)

  @staticmethod
  def _all_gather(pin_layout):
    device = xm.xla_device()
    ordinal = torch.tensor([xm.get_ordinal()], device=device)
    out = xm.all_gather(ordinal, pin_layout=pin_layout)
    xm.mark_step()

    return out.cpu().numpy()

  @parameterized.named_parameters(('pinned', True), ('unpinned', False))
  def test_all_gather(self, pin_layout):
    results = pjrt.run_multiprocess(self._all_gather, pin_layout)

    expected = list(range(len(results)))
    for v in results.values():
      np.testing.assert_array_equal(v, expected)

  @staticmethod
  def _reduce_scatter(pin_layout):
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    tensor = -torch.arange(world_size, dtype=torch.float32).to(device)

    out = xm.reduce_scatter(
        xm.REDUCE_SUM,
        tensor,
        scale=1.0 / world_size,
        scatter_dim=0,
        shard_count=world_size,
        pin_layout=pin_layout,
    )
    xm.mark_step()

    return out.cpu().numpy()

  # 2023-08-02 04:16:36.520884: F external/xla/xla/service/layout_assignment.cc:157] Check failed: ShapeUtil::Compatible(shape_layout.shape(), instruction->operand(operand_no)->shape()) f32[1]{0} is not compatible with f32[2]{0} (for operand 0 of instruction %reduce-scatter.10 = f32[1]{0} reduce-scatter(f32[2]{0} %add.5), replica_groups={}, constrain_layout=true, dimensions={0}, to_apply=%AddComputation.6)
  @parameterized.named_parameters(('pinned', True), ('unpinned', False))
  def test_reduce_scatter(self, pin_layout):
    results = pjrt.run_multiprocess(self._reduce_scatter, pin_layout)

    for ordinal, value in results.items():
      np.testing.assert_array_equal(value, [-ordinal])

  @staticmethod
  def _all_to_all(pin_layout):
    device = xm.xla_device()
    world_size = xm.xrt_world_size()

    tensor = torch.cat(
        [
            -torch.arange(world_size, dtype=torch.float32).view(-1, 1, 1),
            torch.ones(world_size, 1, 1) * xm.get_ordinal(),
        ],
        dim=1,
    ).to(device)
    xm.mark_step()

    out = xm.all_to_all(
        tensor,
        split_dimension=0,
        concat_dimension=2,
        split_count=world_size,
        pin_layout=pin_layout,
    )

    return out.cpu().numpy()

  @parameterized.named_parameters(('pinned', True), ('unpinned', False))
  def test_all_to_all(self, pin_layout):
    results = pjrt.run_multiprocess(self._all_to_all, pin_layout)

    for ordinal, value in results.items():
      np.testing.assert_array_equal(value, [[[-ordinal] * len(results),
                                             list(range(len(results)))]])


if __name__ == '__main__':
  absltest.main()
