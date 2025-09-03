import jax
import torch
import torchax
import unittest

from jax.sharding import NamedSharding, PartitionSpec as P

class ToDeviceTest(unittest.TestCase):

  def test_to_device_twice(self):
      env = torchax.default_env()
      mesh = jax.make_mesh((jax.device_count(), ), ('axis', ))
      with env:
        step1 = torch.ones(
            100,
            100,
        )
        step2 = torch.triu(step1, diagonal=1)
        step3 = step2.to(dtype=torch.bool, device='jax')
        step3.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        print(step3.to('jax'))
        self.assertEqual(step3.device.type, 'jax')


if __name__ == '__main__':
    unittest.main()
