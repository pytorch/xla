from collections.abc import Sequence
import functools
import string
import sys
from typing import Any, Callable, Optional, Union
import weakref

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.experimental.xla_sharding as xs
import torch_xla.utils.utils as xu
import torch_xla.core.xla_env_vars as xenv
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor

try:
  import rich
  import rich.align
  import rich.box
  import rich.console
  import rich.padding
  import rich.style
  import rich.table
  RICH_ENABLED = True
except:
  RICH_ENABLED = False

# Sharding visualization
sharding_callbacks = weakref.WeakValueDictionary()
_INSPECT_SHARDING_CALL_NAME = "InspectSharding"


class ShardingCallbackInfo:

  def __init__(self, callback, module_context):
    self.callback = callback
    self.module_context = module_context


def visualize_sharding(sharding: str,
                       use_color: bool = True,
                       scale: float = 1.,
                       min_width: int = 9,
                       max_width: int = 80):
  """Visualizes a ``Sharding`` using ``rich``.
  Args:
    sharding (`str`): sharding of given tensor with SPMD
    use_color (`bool`): whether use color or not
    scale (`float`): scale of table visualized in console
    min_width (`int`): min width used to setup table to visualize
    max_width (`int`): max width used to setup table to visualize
  Returns:
    table to visualize given tensor sharding. This function
    will also visualize the sharding of the tensor without as return.
  """

  if not RICH_ENABLED:
    raise ValueError("`visualize_sharding` requires `rich` to be installed.")

  slices: dict[tuple[int, ...], set[int]] = {}
  heights: dict[tuple[int, ...], Optional[float]] = {}
  widths: dict[tuple[int, ...], float] = {}

  if len(sharding) > 0:
    # sharding is longer than 0
    # eg: '{devices=[2,2]0,1,2,3}' # 13
    # eg: '{replicated}'
    # eg: '{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}' # 15
    if sharding == '{replicated}':
      # eg: '{replicated}'
      heights = 1
      widths = 1
      num_devices = xr.global_runtime_device_count()
      device_ids = list(range(num_devices))
      slices.setdefault((0, 0), device_ids)
    else:
      # `device_indices_map`: [0, 1, 2, 3]
      # `sharding_spac`: [2, 2]
      sharding_spac = sharding[sharding.index('['):sharding.index(']') + 1]
      if len(sharding) >= 25 and sharding[-24:-1] == 'last_tile_dim_replicate':
        device_list = list(sharding[sharding.index(']') + 1:-24])
        device_indices_map = [int(i) for i in device_list[:-1] if i != ',']
        heights = int(sharding_spac[1])
        widths = int(sharding_spac[3])
        last_dim_depth = int(sharding_spac[5])
        devices_len = len(device_indices_map)
        len_after_dim_down = devices_len // last_dim_depth
        for i in range(len_after_dim_down):
          slices.setdefault(
              (i // widths, i % widths),
              device_indices_map[i * last_dim_depth:(i + 1) * last_dim_depth])
      elif sharding[-1] == "}":
        # eg: '{devices=[2,2]0,1,2,3}' # 13
        device_list = list(sharding[sharding.index(']') + 1:-1])
        device_indices_map = [int(i) for i in device_list if i != ',']
        heights = int(sharding_spac[1])
        widths = int(sharding_spac[3])
        devices_len = len(device_indices_map)
        for i in range(devices_len):
          slices.setdefault((i // widths, i % widths), device_indices_map[i])
      else:
        raise ValueError("sharding is not organized as expected")
  else:
    raise ValueError("sharding has no value")

  num_rows = heights
  num_cols = widths

  console = rich.console.Console(width=max_width)
  use_color = use_color and console.color_system is not None

  base_height = int(3 * scale)
  aspect_ratio = 1
  base_width = int(base_height * aspect_ratio)
  height_to_width_ratio = 1.5

  # eg: '{devices=[2,2]0,1,2,3}' # 13
  # eg: '{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}' # 15

  # slcs is the data we saved on this slice
  # `device_indices_map`: [0, 1, 2, 3]
  # `sharding_spac`: [2, 2]

  pjrt_device = xu.getenv_as(xenv.PJRT_DEVICE, str)
  device_kind = pjrt_device

  table = rich.table.Table(
      show_header=False,
      show_lines=not use_color,
      padding=0,
      highlight=not use_color,
      pad_edge=False,
      box=rich.box.SQUARE if not use_color else None)
  for i in range(num_rows):
    col = []
    for j in range(num_cols):
      entry = f"{device_kind} " + str(slices[i, j])
      width, maybe_height = widths, heights  # widths[i, j], heights[i, j]
      width = int(width * base_width * height_to_width_ratio)
      if maybe_height is None:
        height = 1
      else:
        height = int(maybe_height * base_height)
      width = min(max(width, min_width), max_width)
      left_padding, remainder = divmod(width - len(entry) - 2, 2)
      right_padding = left_padding + remainder
      top_padding, remainder = divmod(height - 2, 2)
      bottom_padding = top_padding + remainder

      color = None
      text_color = None

      padding = (top_padding, right_padding, bottom_padding, left_padding)
      padding = tuple(max(x, 0) for x in padding)

      col.append(
          rich.padding.Padding(
              rich.align.Align(entry, "center", vertical="middle"),
              padding,
              style=rich.style.Style(bgcolor=color, color=text_color)))
    table.add_row(*col)
  console.print(table, end='\n\n')
  return table


def visualize_tensor_sharding(t, **kwargs):
  """Visualizes an array's sharding."""
  if torch.is_tensor(t):
    import torch_xla
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    return visualize_sharding(sharding, **kwargs)
  elif (isinstance(t, XLAShardedTensor)):
    import torch_xla
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t.global_tensor)
    return visualize_sharding(sharding, **kwargs)
