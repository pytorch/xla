"""Flax interop."""

import torch
import torchax as tx
import torchax.interop


class FlaxNNModule(torch.nn.Module):

  def __init__(self, env, flax_module, sample_args, sample_kwargs):
    super().__init__()
    prng = env.prng_key
    self._params = tx.interop.call_jax(flax_module.init, prng, *sample_args, **sample_kwargs)
    self._flax_module = flax_module
    
  def forward(self, *args, **kwargs):
    return tx.interop.call_jax(self._flax_module.apply, self._params, *args, **kwargs)

    