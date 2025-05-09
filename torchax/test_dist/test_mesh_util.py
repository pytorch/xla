import unittest

import jax
from jax.sharding import PartitionSpec
import torch
import torchax
from torchax.mesh_util import Mesh, SingleAxisSharder


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
        len(model.a.weight.jax().addressable_shards), len(jax.devices()))
    self.assertEqual(
        len(model.a.bias.jax().addressable_shards), len(jax.devices()))

  def test_sharder_call(self):
    """Test the __call__ method produces the correct PartitionSpec."""
    sharder = SingleAxisSharder(axis_name="fsdp", axis_size=4)
    # Use a simple named tuple instead of MagicMock
    shaped_type = torch.ones((5, 8, 12))  # Middle dim divisible by 4

    spec = sharder("param_name", shaped_type)
    self.assertEqual(spec, PartitionSpec(None, "fsdp", None))

  def test_sharder_call_no_shardable(self):
    """Test __call__ when no dimension is shardable."""
    sharder = SingleAxisSharder(axis_name="fsdp", axis_size=4)
    shaped_type = torch.ones((5, 7, 11))

    with self.assertRaisesRegex(AssertionError,
                                "Unable to find a dim to shard"):
      sharder("param_name", shaped_type)


if __name__ == "__main__":
  unittest.main()
