import torch
from torch import nn
import torchax
from . import base_test_util


class CustomConv1(torch.nn.Module):

  def __init__(
      self,
      channels_conv1=3,
      width_conv1=3,
      channels_conv2=5,
      width_conv2=5,
      hidden_layer_size=50,
  ):
    super(CustomConv1, self).__init__()
    self.conv1 = nn.Conv1d(1, channels_conv1, width_conv1)
    self.conv2 = nn.Conv1d(channels_conv1, channels_conv2, width_conv2)
    self.fc1 = nn.Linear(hidden_layer_size, 2)

  def forward(self, x):
    x = nn.functional.max_pool1d(nn.functional.relu(self.conv1(x)), 2, stride=2)
    x = nn.functional.max_pool1d(nn.functional.relu(self.conv2(x)), 2, stride=2)
    x = torch.flatten(x, 1)
    x = nn.functional.softmax(self.fc1(x), dim=1)
    return x


class CustomConv2(nn.Module):

  def __init__(self):
    super().__init__()
    inp = 4
    out = 16

    self.conv = nn.Conv2d(inp, out, kernel_size=3, padding=1)

    # This is supposed to be a squeeze and excitation block.
    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    self.scale = nn.Sequential(nn.Linear(out, out), nn.Sigmoid())

  def forward(self, x):
    x = self.conv(x)

    b = x.shape[0]
    ap = self.avg_pool(x).view(b, -1)
    ap = self.scale(ap)
    ap = ap.view(b, -1, 1, 1)

    return x * ap


class ConvTest(base_test_util.TestCase):

  def test_conv1(self):
    env = torchax.default_env()
    m = CustomConv1()
    arg = torch.randn((20, 1, 50))
    res = m(arg)

    jax_weights, jax_func = torchax.extract_jax(m)
    arg = env.t2j_copy(arg)
    res2 = jax_func(jax_weights, (arg,))
    res2_torch = env.j2t_copy(res2)
    self.assertTrue(torch.allclose(res, res2_torch))

  def test_conv2(self):
    env = torchax.default_env()
    m = CustomConv2()
    arg = torch.randn((20, 4, 50, 100))
    res = m(arg)
    jax_weights, jax_func = torchax.extract_jax(m)
    arg = env.t2j_copy(arg)
    res2 = jax_func(jax_weights, (arg,))
    res2_torch = env.j2t_copy(res2)
    self.assertTrue(torch.allclose(res, res2_torch, atol=1e-4, rtol=1e-4))


if __name__ == '__main__':
  base_test_util.main()
