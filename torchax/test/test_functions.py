from typing import Callable
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torchax
import torchax.tensor


class SeqModel(torch.nn.Module):
    """ Architecture is LLM generated """
    def __init__(self):
        super().__init__()
        self.gru = torch.nn.GRU(20, 30, batch_first=True)
        self.linear = torch.nn.Linear(30, 1)

    def forward(self, x: torch.Tensor):
        output, _ = self.gru(x)  #output, hidden state
        output = self.linear(output)
        return output


class TestTorchFunctions(parameterized.TestCase):

  def setUp(self):
    self.env = torchax.tensor.Environment()
    self.env.config.use_torch_native_for_cpu_tensor = False
    torchax.enable_accuracy_mode()

  @parameterized.named_parameters(
      ('tensor_2d', lambda: torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])),
      ('tensor_1d', lambda: torch.tensor([0, 1],)),
      ('tensor_scalar', lambda: torch.tensor(3.14159,)),
      ('tensor_empty', lambda: torch.tensor([],)),
      ('tensor_dtype', lambda: torch.tensor([[0.11111, 0.222222, 0.3333333]],
                                            dtype=torch.float64)),
  )
  def test_tensor_constructor(self, func: Callable[[], torch.Tensor]):
    expected = func()

    with self.env:
      actual = func()
      self.assertIsInstance(actual, torchax.tensor.Tensor)

    torch.testing.assert_close(torchax.tensor.j2t(actual._elem), expected)

  def test_dont_capture_conversion(self):
    t = torch.tensor([1,2,3])
    with self.env:
      t2 = self.env.to_xla(t)
      # assert no exceptions

  def test_brackets(self):
    with self.env:
      a = torch.randn((2,3))
      a[1] = 9
      self.assertEqual(a[1, 0].item(), 9)

  def test_bernoulli_inplace(self):
    with self.env:
      a = torch.randn((2,3))
      a.bernoulli_(0.4)

  def test_flatten(self):
    with self.env:
      a = torch.randn((2,3,4))
      a = a.flatten(0, 1)
      self.assertEqual(tuple(a.shape), (6, 4))

  def test_rnn(self):
    model = SeqModel()
    x = torch.randn((2, 100, 20))
    res = model(x)
    with self.env:
      model.to('jax')
      x = x.to('jax')
      res2 = model(x)
      print(res.shape, res2.shape)

      self.assertEqual(res.shape, res2.shape)

  def test_rms_norm(self):
    model = torch.nn.RMSNorm((100, 20))
    x = torch.randn((2, 100, 20))
    res = model(x)

    with self.env:
      model.to('jax')
      x = x.to('jax')
      res2 = model(x)
      self.assertTrue(
        torch.allclose(res, torchax.tensor.j2t(res2.jax())))





if __name__ == '__main__':
  absltest.main()
