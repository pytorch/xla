"""Flax interop."""

import torch
import torchax as tx
import torchax.interop


class FlaxNNModule(torch.nn.Module):

  def __init__(self, env, flax_module, sample_args, sample_kwargs=None):
    super().__init__()
    prng = env.prng_key
    sample_kwargs = sample_kwargs or {}
    parameter_dict = tx.interop.call_jax(flax_module.init, prng, *sample_args,
                                         **sample_kwargs)

    self._params = self._encode_nested_dict(parameter_dict)

    self._flax_module = flax_module

  def _encode_nested_dict(self, nested_dict):
    child_module = torch.nn.Module()
    for k, v in nested_dict.items():
      if isinstance(v, dict):
        child_module.add_module(k, self._encode_nested_dict(v))
      else:
        child_module.register_parameter(k, torch.nn.Parameter(v))
    return child_module

  def _decode_nested_dict(self, child_module):
    result = dict(child_module.named_parameters(recurse=False))
    for k, v in child_module.named_children():
      result[k] = self._decode_nested_dict(v)
    return result

  def forward(self, *args, **kwargs):
    nested_dict_params = self._decode_nested_dict(self._params)
    return tx.interop.call_jax(self._flax_module.apply, nested_dict_params,
                               *args, **kwargs)
