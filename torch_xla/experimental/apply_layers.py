from typing import Iterable, List, Dict

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from torch_xla.experimental.scan import scan


def apply_layers(layers: Iterable[torch.nn.Module], input_data):
  """Applies each layer in `layers` to `input_data` sequentially.

  `input_data` is provided as input to the first layer in `layers`. The output of one
  layer is provided as input to next layer.

  All modules in `layers` must have the same structure, and they must perform the same
  calculations given the same model parameters and inputs. In practice, this means you
  cannot use different dropout probabilities, parameter shapes, activation functions etc.,
  across the `layers`.

  Under these conditions, this function is equivalent to

    sequential = torch.nn.Sequential(*layers)
    sequential(input_data)

  This function can be faster to compile since it reuses the XLA computation of the
  first layer to perform the computation of all other layers.
  """
  # Handle empty layers case.
  try:
    next(iter(layers))
  except StopIteration:
    return input_data

  # Extract and stack the parameters into a pytree.
  params = [_extract_weights_dict(layer) for layer in layers]
  _ensure_same_structure(params)
  stacked_params = tree_map(lambda *tensors: torch.stack(tensors, dim=0),
                            *params)

  # Use the first layer as the example/template layer.
  from copy import deepcopy
  example_layer = deepcopy(next(iter(layers)))

  # Hollow out the weights and biases in the example layer.
  example_layer = example_layer.to_empty(device=None)

  # Define the function to apply at each step
  def one_layer(carry, params):
    # Apply the current layer's weights and biases to the example layer,
    # then run the resulting layer.
    _apply_weights_dict(example_layer, params)
    # TODO(yifeit): it should be possible to return `None` as opposed to
    # `example_layer(carry) * 0`, for additional clarity. There is no extra
    # computation since we discard `ys` right after.
    return example_layer(carry), example_layer(carry) * 0

  final_carry, _ = scan(one_layer, input_data, stacked_params)

  return final_carry


def _extract_weights_dict(module: nn.Module):
  """
  Extracts the parameters (weights and biases) from a PyTorch module and
  stores them in a dictionary.
  """
  weights_dict = {name: param for name, param in module.named_parameters()}
  return weights_dict


def _ensure_same_structure(params: List[Dict[str, torch.nn.Parameter]]):
  """
  Verifies that all dictionaries in `params` have the same structure:
  they have the same keys and all the values have the same shape.
  """
  if not params:
    return

  reference_keys = set(params[0].keys())
  reference_shapes = {key: params[0][key].shape for key in reference_keys}

  for idx, param_dict in enumerate(params[1:], start=1):
    current_keys = set(param_dict.keys())

    # Check if keys match
    if current_keys != reference_keys:
      missing_keys = reference_keys - current_keys
      extra_keys = current_keys - reference_keys
      error_message = f"Layer {idx} has mismatched set of parameters."
      if missing_keys:
        error_message += f" Missing params: {missing_keys}."
      if extra_keys:
        error_message += f" Extra params: {extra_keys}."
      raise ValueError(error_message)

    # Check if shapes match for each key
    for key in reference_keys:
      ref_shape = reference_shapes[key]
      current_shape = param_dict[key].shape
      if ref_shape != current_shape:
        raise ValueError(
            f"Shape mismatch for parameter '{key}' in layer {idx}: "
            f"expected {ref_shape}, got {current_shape}.")


def _apply_weights_dict(module: nn.Module, weights_dict):
  """
  Re-applies the weights and biases from the dictionary back to the PyTorch module.
  """
  for name, param in module.named_parameters():
    if name in weights_dict:
      torch.utils.swap_tensors(param, weights_dict[name].clone())
