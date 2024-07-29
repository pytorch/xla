import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from torch_xla import runtime as xr
from copy import deepcopy

import sys
import unittest

import test_utils


def _get_partial_states(s):
  dp_size = xr.global_device_count()
  dp_rank = xr.global_ordinal()

  def convert_fn(tensors):
    torch_xla._XLAC._xla_sync_multi(
        tensors, devices=[], wait=True, sync_xla_data=True)
    ret = []
    for t in tensors:
      ret.append(t.chunk(dp_size)[dp_rank].detach().cpu())
    return ret

  def select_fn(v):
    return type(v) == torch.Tensor and xm.is_xla_tensor(v)

  return xm.ToXlaTensorArena(convert_fn, select_fn).transform(s)


class XlaZeRO1Test(test_utils.XlaTestCase):

  @unittest.skipIf(xr.device_type() == 'TPU', "Crash on TPU")
  def test_zero1(self):
    device = xm.xla_device()

    model = nn.Linear(32, 32)
    x = torch.ones((32, 32))
    x.requires_grad = True
    model = model.to(device)
    x = x.to(device)
    y = model(x).sum()
    y.backward()
    xm.mark_step()

    opt1 = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    opt1.step()
    xm.mark_step()

    opt2 = ZeroRedundancyOptimizer(
        model.parameters(),
        torch.optim.SGD,
        lr=0.01,
        momentum=0.9,
        grad_clipping=False)
    opt2.step()
    xm.mark_step()

    s1 = opt1.state_dict()
    s2 = opt2.state_dict()
    self.assertEqual(_get_partial_states(s1['state']), s2['base_state'])

    s1_clone = deepcopy(xm._maybe_convert_to_cpu(s1))
    s2_clone = deepcopy(xm._maybe_convert_to_cpu(s2))

    opt1.load_state_dict(s1)
    opt2.load_state_dict(s2)
    self.assertEqual(
        _get_partial_states(opt1.state_dict()['state']),
        opt2.state_dict()['base_state'])

    # step still runnable
    opt1.step()
    opt2.step()
    xm.mark_step()

    opt1.load_state_dict(s1_clone)
    opt2.load_state_dict(s2_clone)
    xm.mark_step()
    self.assertEqual(
        _get_partial_states(opt1.state_dict()['state']),
        opt2.state_dict()['base_state'])

    # step still runnable
    opt1.step()
    opt2.step()
    xm.mark_step()


def _mp_fn(index):
  device = xm.xla_device()
  if xm.xla_device_hw(device) in ('TPU', 'CUDA'):
    test = unittest.main(exit=False)
    sys.exit(0 if test.result.wasSuccessful() else 1)
  else:
    print(
        'Default device {} is not a TPU or CUDA device'.format(device),
        file=sys.stderr)


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
