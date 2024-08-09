import torch
import torch_xla2.custom_device

def test_tensor_creation():
  t = torch.tensor([0], device="jax:0")

  assert t.numpy() == [0]

def test_basic_op():
  a = torch.tensor([0], device="jax:0")
  b = torch.tensor([2], device="jax:0")

  c = a + b
  assert c.numpy() == [2]

