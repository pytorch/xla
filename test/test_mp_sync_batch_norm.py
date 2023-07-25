from absl.testing import absltest, parameterized
import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.core.functions as xf
from torch_xla import runtime as xr
import torch_xla.debug.metrics as met
import torch_xla.distributed.xla_multiprocessing as xmp

ATOL: float = 1e-4
RTOL: float = 7e-2
LR: float = 1e-3


def run_step(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
  optimizer = torch.optim.SGD(model.parameters(), lr=LR)
  result = model(batch)
  loss = result.sum()
  if batch.device.type == 'xla':
    loss.backward()
    xm.optimizer_step(optimizer)
  else:
    # Scale as we scale within xm.optimizer_step()
    loss = loss / xm.xrt_world_size()
    loss.backward()
    optimizer.step()
    split_size = batch.shape[0] // xm.xrt_world_size()
    result = result.split(split_size, dim=0)[xm.get_ordinal()]

  return result


def assert_stats(a: torch.nn.Module, b: torch.nn.Module):
  assert a.weight.allclose(b.weight, rtol=RTOL, atol=ATOL)
  assert a.bias.allclose(b.bias, rtol=RTOL, atol=ATOL)
  assert a.running_mean.allclose(b.running_mean, rtol=RTOL, atol=ATOL)
  assert a.running_var.allclose(b.running_var, rtol=RTOL, atol=ATOL)


class TestMpSyncBatchNorm(parameterized.TestCase):

  @staticmethod
  def _sync_bn1d_no_channel(rank):
    torch.manual_seed(1)
    bsz = 32
    length = 64
    t_global = torch.rand((xm.xrt_world_size() * bsz, length))

    # XLA SyncBatchNorm
    device = xm.xla_device()
    t_xla = t_global[bsz * rank:bsz * (rank + 1), ...].to(device)
    sbn_xla = xf.SyncBatchNorm(length).to(device)
    result = run_step(sbn_xla, t_xla)

    # CPU BatchNorm
    bn_cpu = torch.nn.BatchNorm1d(length)
    expected = run_step(bn_cpu, t_global)

    cpu_result = result.cpu()
    assert cpu_result.allclose(expected, rtol=RTOL, atol=ATOL)
    assert_stats(sbn_xla.cpu(), bn_cpu)

    xm.rendezvous('sync_bn1d_no_channel_test')
    xm.master_print('sync_bn1d_no_channel_test ok')

  @staticmethod
  def _sync_bn1d_multi_channel(rank):
    torch.manual_seed(1)
    bsz = 64
    features = 20
    length = 128
    t_global = torch.rand((xm.xrt_world_size() * bsz, features, length))

    # XLA SyncBatchNorm
    device = xm.xla_device()
    t_xla = t_global[bsz * rank:bsz * (rank + 1), ...].to(device)
    sbn_xla = xf.SyncBatchNorm(features).to(device)
    result = run_step(sbn_xla, t_xla)

    # CPU BatchNorm
    bn_cpu = torch.nn.BatchNorm1d(features)
    expected = run_step(bn_cpu, t_global)

    cpu_result = result.cpu()
    assert cpu_result.allclose(expected, rtol=RTOL, atol=ATOL)
    assert_stats(sbn_xla.cpu(), bn_cpu)

    xm.rendezvous('sync_bn1d_multi_channel_test')
    xm.master_print('sync_bn1d_multi_channel_test ok')

  @staticmethod
  def _sync_bn2d(rank):
    torch.manual_seed(1)
    bsz = 8
    features = 10
    h, w = 64, 64
    t_global = torch.rand((xm.xrt_world_size() * bsz, features, h, w))

    # XLA SyncBatchNorm
    device = xm.xla_device()
    t_xla = t_global[bsz * rank:bsz * (rank + 1), ...].to(device)
    sbn_xla = xf.SyncBatchNorm(features).to(device)
    result = run_step(sbn_xla, t_xla)

    # CPU BatchNorm
    bn_cpu = torch.nn.BatchNorm2d(features)
    expected = run_step(bn_cpu, t_global)

    cpu_result = result.cpu()
    assert cpu_result.allclose(expected, rtol=RTOL, atol=ATOL)
    assert_stats(sbn_xla.cpu(), bn_cpu)

    xm.rendezvous('sync_bn2d_test')
    xm.master_print('sync_bn2d_test ok')

  @staticmethod
  def _sync_bn3d(rank):
    torch.manual_seed(1)
    bsz = 16
    features = 32
    d, h, w = 16, 32, 32
    t_global = torch.rand((xm.xrt_world_size() * bsz, features, d, h, w))

    # XLA SyncBatchNorm
    device = xm.xla_device()
    t_xla = t_global[bsz * rank:bsz * (rank + 1), ...].to(device)
    sbn_xla = xf.SyncBatchNorm(features).to(device)
    result = run_step(sbn_xla, t_xla)

    # CPU BatchNorm
    bn_cpu = torch.nn.BatchNorm3d(features)
    expected = run_step(bn_cpu, t_global)

    cpu_result = result.cpu()
    assert cpu_result.allclose(expected, rtol=RTOL, atol=ATOL)
    assert_stats(sbn_xla.cpu(), bn_cpu)

    xm.rendezvous('sync_bn3d_test')
    xm.master_print('sync_bn3d_test ok')

  def test_sync_bn1d_no_channel(self):
    xmp.spawn(self._sync_bn1d_no_channel, args=())

  def test_sync_bn1d_multi_channel(self):
    xmp.spawn(self._sync_bn1d_multi_channel, args=())

  def test_sync_bn2d(self):
    xmp.spawn(self._sync_bn2d, args=())

  def test_sync_bn3d(self):
    xmp.spawn(self._sync_bn3d, args=())


if __name__ == '__main__':
  absltest.main()
