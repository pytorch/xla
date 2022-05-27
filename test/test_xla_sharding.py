import os
import sys

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor
import unittest
import numpy as np

# A placeholder for future sharding tests


class XlaShardingTest(unittest.TestCase):

  def test_xla_sharded_tensor(self):
    # Test XLAShardedTensor basic properties

    # Simple 1-D sharding
    num_devices = len(xm.get_xla_supported_devices("TPU"))
    mesh_shape = (1, num_devices)
    partition_spec = (1,)
    t1 = torch.tensor([2.0, 3.0], dtype=torch.float, device=xm.xla_device())
    t1_sharded = XLAShardedTensor(t1, mesh_shape, partition_spec)
    t2 = torch.tensor([2.0, 3.0], dtype=torch.float, device=xm.xla_device())
    t3 = torch.add(t1_sharded, t2)

    assert isinstance(
        t3, XLAShardedTensor), "Sharded ops should return XLAShardedTensor."
    assert t3.size() == t1.size(
    ), "Sharded output should return unpartitioned tensor size."

    device_ids = np.array(range(num_devices))
    device_assignment = list(device_ids.reshape(mesh_shape))
    # TODO(yeounoh) sharding_spec is broken. The intention is to
    # check if the returned sharding spec holds the correct device
    # assignment as intended.
    # assert device_assignment == t1_sharded.sharding_spec[0]


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
