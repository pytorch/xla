
    
import unittest
import torch
from torch.utils import _pytree as pytree

from torchax import (
    interop,
    mesh_util,
    tensor
)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)
    

class TestTensorStateDict(unittest.TestCase):
    def test_load_statedict(self):
        mesh = mesh_util.Mesh.fsdp_mesh()
        model = mesh.initialize_model_sharded(Model, ())
        model = interop.JittableModule(model)        
        state_dict = model.cpu_state_dict()
        is_xla_tensor = pytree.tree_map(
            lambda t: isinstance(t, tensor.Tensor),
            state_dict
        )
        assert not any(is_xla_tensor.values()), "State dict should not contain XLA tensors"

if __name__ == '__main__':
    unittest.main()