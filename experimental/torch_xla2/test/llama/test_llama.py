import unittest
import jax
import torch
from torch._functorch.make_functional import make_functional_with_buffers
from torch_xla2 import tensor, ops  # pylint: disable=unused-import

from .. import test_base
from . import llama_model
from torch.utils import _pytree as pytree


class LlamaTest(test_base.TestCase):

  def test_can_run(self):
    sample_args = (
        torch.randint(0, 32000, (1, 2048)),
        torch.arange(0, 2048),
    )
    sample_args = pytree.tree_map(tensor.move_to_device, sample_args)

    model_args = llama_model.ModelArgs(
        block_size=2048,
        vocab_size=32000,
        n_layer=2,
        n_head=4,
        dim=256,
    )
    m = llama_model.Transformer(model_args)
    m.to(torch.bfloat16)
    m.setup_caches(1, 2048)
    m_func, weights, buffer = make_functional_with_buffers(m)

    causal_mask = tensor.move_to_device(m.causal_mask)
    freqs_cis = tensor.move_to_device(m.freqs_cis)
    weights = pytree.tree_map(tensor.move_to_device, weights)
    buffer = pytree.tree_map(tensor.move_to_device, buffer)

    @jax.jit
    def m_func_jit(weights, buffer, args, causal_mask, freqs_cis):
      weights, buffer, args, causal_mask, freqs_cis = tensor.wrap(
          (weights, buffer, args, causal_mask, freqs_cis)
      )
      m_func.stateless_model.freqs_cis = freqs_cis
      m_func.stateless_model.causal_mask = causal_mask
      res = m_func(weights, buffer, *args)
      res = tensor.unwrap(res)
      return res

    args = weights, buffer, sample_args, causal_mask, freqs_cis
    args = tensor.unwrap(args)
    # print(m_func_jit.lower(*args).as_text())
    res = m_func_jit(*args)
    jax.block_until_ready(res)


if __name__ == "__main__":
  test_base.main()
