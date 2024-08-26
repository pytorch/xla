import sys
import torch
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
from torch_xla.distributed.spmd import Mesh
import numpy as np
import torch_xla.distributed.spmd as xs
import torch_xla.distributed.parallel_loader as pl
import unittest

xr.use_spmd()

class MpInputShardingTest(unittest.TestCase):
    class fake_dataloader:
        def __init__(self, batch):
            self.batch = batch

        def __iter__(self):
            return self
        
        def __next__(self):
            return self.batch

    @unittest.skipUnless(xr.global_runtime_device_count() > 1,
                       "Multiple devices required for tupled partition spec")
    def test_multiple_inputs(self):
        device = xm.xla_device()
        batch = {'x': torch.randn((16, 128)), 'y': torch.randn((16, 128, 128))}
        train_loader = self.fake_dataloader(batch)
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (num_devices, 1)
        device_ids = np.arange(num_devices)
        mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))
        
        train_loader = pl.MpDeviceLoader(
            train_loader, device, input_sharding={'x': xs.ShardingSpec(mesh, ('x', None)), 'y': xs.ShardingSpec(mesh, ('x', None, None))})
        train_loader = iter(train_loader)
        data = next(train_loader)
        annotation_x = '{devices=[%d,1]%s}' % (
            num_devices, ','.join(
                [str(i) for i in range(num_devices)]))
        annotation_y = '{devices=[%d,1,1]%s}' % (
            num_devices, ','.join(
                [str(i) for i in range(num_devices)]))
        self.assertEqual(annotation_x, torch_xla._XLAC._get_xla_sharding_spec(data['x']))
        self.assertEqual(annotation_y, torch_xla._XLAC._get_xla_sharding_spec(data['y']))

    @unittest.skipUnless(xr.global_runtime_device_count() > 1,
                       "Multiple devices required for tupled partition spec")
    def test_single_tensor(self):
        device = xm.xla_device()
        batch = torch.randn((16, 128))
        train_loader = self.fake_dataloader(batch)
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (num_devices, 1)
        device_ids = np.arange(num_devices)
        mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))
        
        train_loader = pl.MpDeviceLoader(
            train_loader, device, input_sharding=xs.ShardingSpec(mesh, ('x', None)))
        train_loader = iter(train_loader)
        data = next(train_loader)
        annotation = '{devices=[%d,1]%s}' % (
            num_devices, ','.join(
                [str(i) for i in range(num_devices)]))
        self.assertEqual(annotation, torch_xla._XLAC._get_xla_sharding_spec(data))

    @unittest.skipUnless(xr.global_runtime_device_count() > 1,
                       "Multiple devices required for tupled partition spec")
    def test_error_missing_keys(self):
        device = xm.xla_device()
        # if xm.xla_device_hw(device) == 'TPU':
        batch = {'x': torch.randn((16, 128)), 'y': torch.randn((16, 128, 128))}
        train_loader = self.fake_dataloader(batch)
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (num_devices, 1)
        device_ids = np.arange(num_devices)
        mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))
        train_loader = pl.MpDeviceLoader(
            train_loader, device, input_sharding={'x': xs.ShardingSpec(mesh, ('x', None))})
        train_loader = iter(train_loader)
        with self.assertRaises(KeyError):
            data = next(train_loader)

if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)