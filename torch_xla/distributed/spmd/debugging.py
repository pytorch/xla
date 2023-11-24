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


Color = Union[tuple[float, float, float], str]
ColorMap = Callable[[float], tuple[float, float, float, float]]


def _canonicalize_color(color: Color) -> str:
  if isinstance(color, str):
    return color
  r, g, b = (int(a * 255) for a in color)
  return f"#{r:02X}{g:02X}{b:02X}"


def _get_text_color(color: str) -> str:
  r, g, b = torch.map(lambda x: int(x, 16),
                      (color[1:3], color[3:5], color[5:7]))
  if (r * 0.299 + g * 0.587 + b * 0.114) > 186:
    return "#000000"
  return "#ffffff"


def make_color_iter(color_map, num_rows, num_cols):
  num_colors = num_rows * num_cols
  color_values = np.linspace(0, 1, num_colors)
  idx = 0
  for _ in range(num_colors):
    yield color_map(color_values[idx])
    idx = (idx + num_colors // 2 + bool(num_colors % 2 == 0)) % num_colors


def visualize_sharding(shape: torch.Size,
                       sharding: str,
                       use_color: bool = True,
                       scale: float = 1.,
                       min_width: int = 9,
                       max_width: int = 80,
                       color_map: Optional[ColorMap] = None):
  """Visualizes a ``Sharding`` using ``rich``."""
  if not RICH_ENABLED:
    raise ValueError("`visualize_sharding` requires `rich` to be installed.")

  if len(shape) > 2 or len(shape) < 1:
    raise ValueError(
        "`visualize_sharding` only works for shapes with 1 and 2 dimensions.")

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
  if use_color and not color_map:
    try:
      import matplotlib as mpl
      color_map = mpl.colormaps["tab20b"]
    except ModuleNotFoundError:
      use_color = False

  base_height = int(3 * scale)
  aspect_ratio = (shape[1] if len(shape) == 2 else 1) / shape[0]
  base_width = int(base_height * aspect_ratio)
  height_to_width_ratio = 1.5

  # eg: '{devices=[2,2]0,1,2,3}' # 13
  # eg: '{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}' # 15

  # slcs is the data we saved on this slice
  # `device_indices_map`: [0, 1, 2, 3]
  # `sharding_spac`: [2, 2]

  # set the device kind to TPU as default since `sharding` here is `str`, TODO(@manfei): get device kind from commands for TPU/GPU/CPU
  device_kind = 'TPU'  # next(iter(sharding.device_set)).platform.upper()

  color_iter = make_color_iter(color_map, num_rows, num_cols)
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

      if use_color:
        color = _canonicalize_color(next(color_iter)[:3])
        text_color = _get_text_color(color)
        top_padding += 1
        bottom_padding += 1
        left_padding += 1
        right_padding += 1
      else:
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


def visualize_tensor_sharding(ter, **kwargs):
  """Visualizes an array's sharding."""
  import torch_xla
  sharding = torch_xla._XLAC._get_xla_sharding_spec(ter)
  return visualize_sharding(ter.shape, sharding, **kwargs)
