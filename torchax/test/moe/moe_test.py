import torchax
import torchax.interop
import torch
import unittest
import jax

from test.moe import model


class TestMoe(unittest.TestCase):

  def _make_tiny_config(self):
    return model.ModelArgs(
        block_size=128,
        vocab_size=32000,
        n_layer=4,
        n_head=4,
        dim=128,
        intermediate_size=None,
        n_local_heads=-1,
        head_dim=32,
        rope_base=10000,
        norm_eps=1e-5,
        num_experts=8,
        num_activated_experts=2,
    )

  def _random_init(self, model):
    new_state_dict = {}

    for k, v in model.state_dict().items():
      new_state_dict[k] = torch.randn_like(v)

    model.load_state_dict(new_state_dict, assign=True)
    return model

  def test_moe_layer(self):
    model_args = self._make_tiny_config()

    moe_layer = model.MOEFeedForward(model_args)
    moe_layer = self._random_init(moe_layer)
    seqlen = 32
    x = torch.randn((seqlen, model_args.dim))
    res = moe_layer(x)

    env = torchax.default_env()
    model_xla = env.to_xla(moe_layer)
    x_xla = env.to_xla(x)
    with jax.default_matmul_precision('float32'):
      res_xla = model_xla(x_xla)
    res2 = res_xla.to('cpu')
    print('max diff', torch.max((res - res2).abs()))

    self.assertTrue(torch.allclose(res2, res, atol=1e-2))

    # test can jit

    def f(weights, x):
      return torch.func.functional_call(moe_layer, weights, (x,))

    fjitted = torchax.interop.jax_jit(f)
    weights_xla = env.to_xla(moe_layer.state_dict())

    print(fjitted(weights_xla, x_xla))


if __name__ == '__main__':
  unittest.main()
