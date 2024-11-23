from typing import Iterable, Mapping, Sequence, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map
from functorch.compile import default_partition

from torch_xla.experimental.scan import scan


def scan_layers(layers: Iterable[torch.nn.Module],
                input_data,
                partition_fn=default_partition):
  """Runs each layer in `layers` sequentially, starting with `input_data`.
  
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
  
  Args:
    layers: (Iterable[torch.nn.Module]) A list of layers to run.

    input_data: The input to be given to the first layer from `layers`.

    partition_fn: (Optional[Callable]) The graph parition function passed to AOTAutograd.
      Since this function uses AOTAutograd to trace `fn`, you may override what computation
      happen in the forward and backward passes by specifying different partition functions.
      `default_partition` implies no activation checkpointing. You may specify
      `functorch.compile.min_cut_rematerialization_partition` to use min-cut based
      activation checkpointing. You may also write your own partitioner to insert any custom
      logic such as host offloading of activations.
      
  Returns:
    The output of the last layer from `layers`.
    
  Example:

    >>> import torch_xla.runtime
    >>> import torch
    >>> import torch.nn as nn
    >>> from torch_xla.experimental.scan_layers import scan_layers
    >>> with torch_xla.runtime.xla_device():
    >>>   layers = [nn.Linear(16, 16) for i in range(10)]
    >>>   input = torch.randn(16)
    >>> output = scan_layers(layers, input)
    >>> assert output.shape == (16,)  # Output is the 10-th layer output
    >>> print(output)  # Some random numbers
  """
  # Handle empty layers case.
  try:
    first_layer = next(iter(layers))
  except StopIteration:
    return input_data

  # Extract and stack the parameters and buffers into pytrees.
  params_and_buffers = [_extract_weights_and_buffers(layer) for layer in layers]
  params_list = [p for p, _ in params_and_buffers]
  buffers_list = [b for _, b in params_and_buffers]

  _ensure_same_structure(params_list)
  _ensure_same_structure(buffers_list)

  stacked_params = tree_map(lambda *tensors: torch.stack(tensors, dim=0),
                            *params_list)
  stacked_buffers = tree_map(lambda *tensors: torch.stack(tensors, dim=0),
                             *buffers_list)

  # Use the first layer as the example/template layer.
  from copy import deepcopy
  example_layer = deepcopy(first_layer)

  # Define the function to apply at each step
  def one_layer(carry, params_buffers):
    # Apply the current layer's weights and biases to the example layer,
    # then run the resulting layer.
    output = torch.func.functional_call(  # type: ignore
        example_layer, params_buffers, carry, strict=True)
    return output, None

  stacked_params_buffers = (stacked_params, stacked_buffers)
  final_carry, _ = scan(
      one_layer, input_data, stacked_params_buffers, partition_fn=partition_fn)

  return final_carry


def _extract_weights_and_buffers(
    module: nn.Module
) -> Tuple[Dict[str, torch.nn.Parameter], Dict[str, torch.Tensor]]:
  """
  Extracts the parameters and buffers from a PyTorch module and
  stores them in separate dictionaries.
  """
  weights_dict = {name: param for name, param in module.named_parameters()}
  buffers_dict = {name: buffer for name, buffer in module.named_buffers()}
  return weights_dict, buffers_dict


def _ensure_same_structure(dicts: Sequence[Mapping[str, torch.Tensor]]):
  """
  Verifies that all dictionaries in `dicts` have the same structure:
  they have the same keys and all the values have the same shape.
  """
  if not dicts:
    return

  reference_keys = set(dicts[0].keys())
  reference_shapes = {key: dicts[0][key].shape for key in reference_keys}

  for idx, current_dict in enumerate(dicts[1:], start=1):
    current_keys = set(current_dict.keys())

    # Check if keys match
    if current_keys != reference_keys:
      missing_keys = reference_keys - current_keys
      extra_keys = current_keys - reference_keys
      error_message = f"Layer {idx} has mismatched keys."
      if missing_keys:
        error_message += f" Missing keys: {missing_keys}."
      if extra_keys:
        error_message += f" Extra keys: {extra_keys}."
      raise ValueError(error_message)

    # Check if shapes match for each key
    for key in reference_keys:
      ref_shape = reference_shapes[key]
      current_shape = current_dict[key].shape
      if ref_shape != current_shape:
        raise ValueError(f"Shape mismatch for '{key}' in layer {idx}: "
                         f"expected {ref_shape}, got {current_shape}.")
