import sys
import numpy as np
import unittest

import torch
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
from torch_xla.distributed.spmd import Mesh
import torch_xla.distributed.spmd as xs
import torch_xla.distributed.parallel_loader as pl

xr.use_spmd()


class MpInputShardingTest(unittest.TestCase):

  class fake_dataloader:

    def __init__(self, batch, size=1):
      self.batch = batch
      self.batch_size = size
      self.counter = 0

    def __iter__(self):
      return self

    def __next__(self):
      if self.counter < self.batch_size:
        self.counter += 1
        return self.batch
      raise StopIteration

  @unittest.skipUnless(xr.global_runtime_device_count() > 1,
                       "Multiple devices required for tupled partition spec")
  def test_multiple_inputs(self):
    device = xm.xla_device()
    batch = {'x': torch.randn((16, 128)), 'y': torch.randn((16, 128, 128))}
    train_loader = self.fake_dataloader(batch)
    num_devices = xr.global_runtime_device_count()
    mesh = xs.get_1d_mesh('x')

    train_loader = pl.MpDeviceLoader(
        train_loader,
        device,
        input_sharding={
            'x': xs.ShardingSpec(mesh, ('x', None)),
            'y': xs.ShardingSpec(mesh, ('x', None, None))
        })
    train_loader = iter(train_loader)
    data = next(train_loader)
    annotation_x = '{devices=[%d,1]%s}' % (num_devices, ','.join(
        [str(i) for i in range(num_devices)]))
    annotation_y = '{devices=[%d,1,1]%s}' % (num_devices, ','.join(
        [str(i) for i in range(num_devices)]))
    self.assertEqual(annotation_x,
                     torch_xla._XLAC._get_xla_sharding_spec(data['x']))
    self.assertEqual(annotation_y,
                     torch_xla._XLAC._get_xla_sharding_spec(data['y']))

  @unittest.skipUnless(xr.global_runtime_device_count() > 1,
                       "Multiple devices required for tupled partition spec")
  def test_single_tensor(self):
    device = xm.xla_device()
    batch = torch.randn((16, 128))
    train_loader = self.fake_dataloader(batch)
    num_devices = xr.global_runtime_device_count()
    mesh = xs.get_1d_mesh('x')

    train_loader = pl.MpDeviceLoader(
        train_loader, device, input_sharding=xs.ShardingSpec(mesh, ('x', None)))
    train_loader = iter(train_loader)
    data = next(train_loader)
    annotation = '{devices=[%d,1]%s}' % (num_devices, ','.join(
        [str(i) for i in range(num_devices)]))
    self.assertEqual(annotation, torch_xla._XLAC._get_xla_sharding_spec(data))

  @unittest.skipUnless(xr.global_runtime_device_count() > 1,
                       "Multiple devices required for tupled partition spec")
  def test_error_single_tensor_with_input_sharding_dict(self):
    device = xm.xla_device()
    batch = torch.randn((16, 128))
    train_loader = self.fake_dataloader(batch)
    num_devices = xr.global_runtime_device_count()
    mesh = xs.get_1d_mesh('x')

    train_loader = pl.MpDeviceLoader(
        train_loader,
        device,
        input_sharding={'x': xs.ShardingSpec(mesh, ('x', None))})
    train_loader = iter(train_loader)
    with self.assertRaises(ValueError):
      data = next(train_loader)

  @unittest.skipUnless(xr.global_runtime_device_count() > 1,
                       "Multiple devices required for tupled partition spec")
  def test_input_sharding_none(self):
    device = xm.xla_device()
    batch = {'x': torch.randn((16, 128)), 'y': torch.randn((16, 128, 128))}
    train_loader = self.fake_dataloader(batch)
    num_devices = xr.global_runtime_device_count()

    train_loader = pl.MpDeviceLoader(train_loader, device, input_sharding=None)
    train_loader = iter(train_loader)
    data = next(train_loader)
    annotation = '{replicated}'
    self.assertEqual(annotation,
                     torch_xla._XLAC._get_xla_sharding_spec(data['x']))
    self.assertEqual(annotation,
                     torch_xla._XLAC._get_xla_sharding_spec(data['y']))

  @unittest.skipUnless(xr.global_runtime_device_count() > 1,
                       "Multiple devices required for tupled partition spec")
  def test_error_missing_keys(self):
    device = xm.xla_device()
    batch = {'x': torch.randn((16, 128)), 'y': torch.randn((16, 128, 128))}
    train_loader = self.fake_dataloader(batch)
    mesh = xs.get_1d_mesh('x')
    train_loader = pl.MpDeviceLoader(
        train_loader,
        device,
        input_sharding={'x': xs.ShardingSpec(mesh, ('x', None))})
    train_loader = iter(train_loader)
    with self.assertRaises(KeyError):
      data = next(train_loader)

  @unittest.skipUnless(xr.global_runtime_device_count() > 1,
                       "Multiple devices required for tupled partition spec")
  def test_input_sharding_not_dict(self):
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    batch = {'x': torch.randn((16, 128)), 'y': torch.randn((16, 128))}
    train_loader = self.fake_dataloader(batch)
    mesh = xs.get_1d_mesh('x')
    train_loader = pl.MpDeviceLoader(
        train_loader, device, input_sharding=xs.ShardingSpec(mesh, ('x', None)))
    train_loader = iter(train_loader)
    data = next(train_loader)
    annotation_x = '{devices=[%d,1]%s}' % (num_devices, ','.join(
        [str(i) for i in range(num_devices)]))
    annotation_y = '{devices=[%d,1]%s}' % (num_devices, ','.join(
        [str(i) for i in range(num_devices)]))
    self.assertEqual(annotation_x,
                     torch_xla._XLAC._get_xla_sharding_spec(data['x']))
    self.assertEqual(annotation_y,
                     torch_xla._XLAC._get_xla_sharding_spec(data['y']))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)