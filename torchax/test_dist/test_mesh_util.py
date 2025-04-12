import unittest

import jax
import torch
import torchax
from torchax.mesh_util import Mesh


class MeshUtilTest(unittest.TestCase):
  def setUp(self):
    torchax.enable_globally()

  def test_init_module_sharded(self):
    class TestModule(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(8, 8)

    mesh = Mesh.fsdp_mesh()

    model = mesh.initialize_model_sharded(TestModule, ())
    self.assertEqual(
      len(model.a.weight.jax().addressable_shards), len(jax.devices())
    )
    self.assertEqual(
      len(model.a.bias.jax().addressable_shards), len(jax.devices())
    )


if __name__ == "__main__":
  unittest.main()
