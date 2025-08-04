"""Flax interop."""

import torch
import torchax as tx
import torchax.interop


class FlaxNNModule(torch.nn.Module):
  """A `torch.nn.Module` that wraps a Flax module for interoperability.

  This class allows you to use a Flax module within a PyTorch model. It
  initializes the Flax module, extracts its parameters, and wraps them in a
  `torch.nn.ParameterDict` so they can be managed by PyTorch. The `forward`
  pass then calls the Flax module's `apply` method with the appropriate
  parameters.

  **Attributes:**

  *   `_params` (`torch.nn.Module`): A nested `torch.nn.Module` that holds the
      parameters of the Flax module.
  *   `_flax_module`: The original Flax module.
  """

  def __init__(self, env, flax_module, sample_args, sample_kwargs=None):
    """Initializes the `FlaxNNModule`.

    **Args:**

    *   `env`: The `torchax` environment.
    *   `flax_module`: The Flax module to wrap.
    *   `sample_args`: A tuple of sample arguments to initialize the Flax module.
    *   `sample_kwargs` (optional): A dictionary of sample keyword arguments to
        initialize the Flax module.
    """
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
    """Performs the forward pass by calling the wrapped Flax module."""
    nested_dict_params = self._decode_nested_dict(self._params)
    return tx.interop.call_jax(self._flax_module.apply, nested_dict_params,
                               *args, **kwargs)
