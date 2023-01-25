import os
import queue

from absl.testing import absltest, parameterized

import torch
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_env_vars as xenv
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.experimental import pjrt


class TestExperimentalPjrtMultiCpu(parameterized.TestCase):

  def setUp(self):
    pjrt.set_device_type('CPU')

    os.environ.update({
        xenv.PJRT_CPU_ASYNC_CLIENT: 'true',
        xenv.CPU_NUM_DEVICES: '4',
    })

  def test_default_cpu_device(self):
    os.environ.pop(xenv.CPU_NUM_DEVICES, None)
    os.environ.pop(xenv.PJRT_CPU_ASYNC_CLIENT, None)

    expected = {0: torch.device('xla:0')}
    devices_per_process = pjrt._run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices_per_process, expected)

  def test_multi_cpu_devices(self):
    expected = {
        0: torch.device('xla:0'),
        1: torch.device('xla:1'),
        2: torch.device('xla:2'),
        3: torch.device('xla:3'),
    }

    devices_per_process = pjrt._run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices_per_process, expected)

  @parameterized.named_parameters(('xla_model', xm.get_ordinal),
                                  ('pjrt', pjrt.global_ordinal))
  def test_global_ordinal(self, ordinal_func):
    results = pjrt._run_multiprocess(ordinal_func)
    self.assertListEqual(sorted(results.values()), [0, 1, 2, 3])

  @parameterized.named_parameters(('xla_model', xm.get_local_ordinal),
                                  ('pjrt', pjrt.local_ordinal))
  def test_local_ordinal(self, ordinal_func):
    # TODO(wcromar): add multiprocess tests
    results = pjrt._run_multiprocess(ordinal_func)
    self.assertListEqual(sorted(results.values()), [0, 1, 2, 3])

  @staticmethod
  def _multi_cpu_backwards():
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

  def test_multi_cpu_backwards(self):
    os.environ.update({
        xenv.PJRT_CPU_ASYNC_CLIENT: 'true',
        xenv.CPU_NUM_DEVICES: '4',
    })

    expected = {
        i: {
            'forward_ordinal': i,
            'backward_ordinal': i,
            'device': f'xla:{i}'
        } for i in range(4)
    }
    results = pjrt._run_multiprocess(self._multi_cpu_backwards)

    self.assertDictEqual(results, expected)

  @staticmethod
  def _spawn(index: int, queue: queue.Queue):
    queue.put(index)

  @parameterized.named_parameters(('xmp', xmp.spawn), ('pjrt', pjrt.spawn))
  def test_spawn(self, spawn):
    manager = torch.multiprocessing.Manager()
    queue = manager.Queue(4)
    spawn(self._spawn, args=(queue,))

    indices = sorted(queue.get(block=False) for _ in range(queue.qsize()))
    self.assertListEqual(indices, list(range(4)))

  @staticmethod
  def _hlo_dump(tmpdir: str):
    os.environ['XLA_SAVE_TENSORS_FMT'] = 'hlo'
    os.environ['XLA_SAVE_TENSORS_FILE'] = os.path.join(tmpdir, 'save.hlo')

    x = torch.randn((3, 3), device=xm.xla_device())
    xm.mark_step()
    x.cpu()

  def test_hlo_dump(self):
    tmpdir = self.create_tempdir().full_path
    pjrt._run_multiprocess(self._hlo_dump, tmpdir)

    files = os.listdir(tmpdir)
    for i in range(4):
      self.assertIn(f'save.hlo.{i}', files)


if __name__ == '__main__':
  absltest.main()
