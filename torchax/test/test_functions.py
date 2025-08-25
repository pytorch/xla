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
    torchax.enable_globally()
    torchax.enable_accuracy_mode()
    self.env = torchax.default_env()

  @parameterized.named_parameters(
      ('tensor_2d', [[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]),
      ('tensor_1d', [0, 1]), ('tensor_scalar', 3.14159), ('tensor_empty', []),
      ('tensor_dtype', [[0.11111, 0.222222, 0.3333333]], {
          'dtype': torch.float64
      }))
  def test_tensor_constructor(self, arg, kwargs=None):
    kwargs = kwargs or {}
    expected = torch.tensor(arg, **kwargs)

    actual = torch.tensor(arg, device='jax', **kwargs)
    self.assertIsInstance(actual, torchax.tensor.Tensor)

    torch.testing.assert_close(actual.to('cpu'), expected)

  def test_dont_capture_conversion(self):
    t = torch.tensor([1, 2, 3])
    with self.env:
      t2 = self.env.to_xla(t)
      # assert no exceptions

  def test_brackets(self):
    with self.env:
      a = torch.randn((2, 3))
      a[1] = 9
      self.assertEqual(a[1, 0].item(), 9)

  def test_bernoulli_inplace(self):
    with self.env:
      a = torch.randn((2, 3))
      a.bernoulli_(0.4)

  def test_flatten(self):
    with self.env:
      a = torch.randn((2, 3, 4))
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
      self.assertTrue(torch.allclose(res, res2.to('cpu')))

  @parameterized.named_parameters(
      ('ones', torch.ones, ((2, 2),)), ('zeros', torch.zeros, ((2, 2),)),
      ('empty', torch.empty,
       ((2, 2),)), ('empty_strided', torch.empty_strided,
                    ((2, 2), (2, 1))), ('tensor', torch.tensor, ([2.0, 2.0],)),
      ('eye', torch.eye, (2,)), ('randn', torch.randn, ((2, 2),)),
      ('rand', torch.rand, ((2, 2),)), ('full', torch.full, ((2, 2), 0)))
  def test_requires_grad(self, func, args):
    x = func(*args, requires_grad=True, device='jax')
    self.assertEqual(x.requires_grad, True)


if __name__ == '__main__':
  absltest.main()
