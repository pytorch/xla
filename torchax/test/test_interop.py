import torch
import unittest
from torchax import interop


class M1(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.x = torch.ones(10, 10)


class M(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.a = torch.nn.Linear(100, 100)
    self.b = torch.nn.Parameter(torch.ones(10, 10))
    c = torch.ones(10, 10)
    self.register_buffer('c', c)
    self.register_buffer('c2', c, persistent=False)
    self.d = torch.ones(10, 10)
    self.m1 = M1()


class InteropTest(unittest.TestCase):

  def test_mod_attr(self):
    m = M()
    params, buffers = interop.extract_all_buffers(m)
    self.assertEqual(set(params.keys()), {'a.weight', 'a.bias', 'b'})
    self.assertEqual(set(buffers.keys()), {'c', 'c2', 'd', 'm1.x'})

    interop.set_all_buffers(m, {'a.weight': torch.tensor([0.0])},
                            {'m1.x': torch.tensor([0.0])})
    self.assertEqual(m.a.weight.item(), 0)
    self.assertEqual(m.m1.x.item(), 0)


if __name__ == '__main__':
  unittest.main()
