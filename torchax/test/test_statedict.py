import unittest
import torch
from torch.utils import _pytree as pytree

from torchax import (interop, mesh_util, tensor)


class Model(torch.nn.Module):

  def __init__(self):
    super(Model, self).__init__()
    self.linear = torch.nn.Linear(10, 5)

  def forward(self, x):
    return self.linear(x)


mesh = mesh_util.Mesh.fsdp_mesh()
model = interop.JittableModule(mesh.initialize_model_sharded(Model, ()))


class TestTensorStateDict(unittest.TestCase):

  def test_get_statedict(self):
    state_dict_cpu = model.cpu_state_dict()
    is_xla_tensor = pytree.tree_map(lambda t: isinstance(t, tensor.Tensor),
                                    state_dict_cpu)
    assert not any(
        is_xla_tensor.values()), "State dict should not contain XLA tensors"

  def test_load_statedict(self):
    state_dict_cpu = model.cpu_state_dict()
    state_dict_cpu = pytree.tree_map(torch.zeros_like, state_dict_cpu)
    model.load_state_dict(state_dict_cpu)
    is_zeros = pytree.tree_map(lambda t: torch.equal(t, torch.zeros_like(t)),
                               state_dict_cpu)
    assert all(is_zeros.values()), "State dict should be zeros"

  def test_load_statedict_partial(self):
    state_dict_cpu = model.cpu_state_dict()
    del state_dict_cpu['_model.linear.bias']
    state_dict_cpu = pytree.tree_map(torch.ones_like, state_dict_cpu)
    key_check = model.load_state_dict(state_dict_cpu, strict=False)
    assert key_check.missing_keys == [
        '_model.linear.bias'
    ], "Missing keys should be '_model.linear.bias'"
    linear_weight = model.state_dict()['_model.linear.weight']
    assert torch.equal(
        linear_weight,
        torch.ones_like(linear_weight)), "Linear weight should be ones"


if __name__ == '__main__':
  unittest.main()
