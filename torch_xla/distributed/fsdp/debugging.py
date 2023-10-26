from collections.abc import Sequence # python # 集合抽象基类
import functools # python # 调用高阶其他定义函数
import string # python # string
import sys # python # sys env
from typing import Any, Callable, Optional, Union
# Any是与任意类型匹配, Callable是int到str的函数, Optional是可选, Union是联合型【x，y】表示x或y
import weakref # python # 对象的弱引用

import numpy as np # numpy数字计算
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor
from torch_xla.experimental.xla_sharding import Mesh

# import jax.numpy as jnp # jax的numpy数字计算
# from jax import lax # jax的基元库，支持jax。numpy

# from jax._src import core
# from jax._src import effects
# from jax._src import linear_util as lu
# from jax._src import mesh as mesh_lib
# from jax._src import sharding_impls
# from jax._src import tree_util # 对于pytree数据结构的函数，用户定义数据结构转为jax code
# from jax._src import util
# from jax._src.interpreters import ad
# from jax._src.interpreters import batching
# from jax._src.interpreters import mlir
# from jax._src.interpreters import partial_eval as pe
# from jax._src.lib import xla_client as xc
# from jax._src.lib.mlir import ir
# from jax._src.lib.mlir.dialects import hlo
# from jax._src.sharding import Sharding
# from jax._src.sharding_impls import NamedSharding, parse_flatten_op_sharding

# pytype: disable=import-error
try:
  import rich
  import rich.align #  空格对齐
  import rich.box # 画框
  import rich.console # 获取控制台结果
  import rich.padding # 填充
  import rich.style # 文本颜色，字体样式
  import rich.table # 列
  RICH_ENABLED = True 
except:
  RICH_ENABLED = False
# pytype: enable=import-error

# class DebugEffect(effects.Effect):
#   __str__ = lambda self: "Debug"
# debug_effect = DebugEffect()

# class OrderedDebugEffect(effects.Effect):
#   __str__ = lambda self: "OrderedDebug"
# ordered_debug_effect = OrderedDebugEffect()

# effects.ordered_effects.add_type(OrderedDebugEffect)
# effects.lowerable_effects.add_type(DebugEffect)
# effects.lowerable_effects.add_type(OrderedDebugEffect)
# effects.control_flow_allowed_effects.add_type(DebugEffect)
# effects.control_flow_allowed_effects.add_type(OrderedDebugEffect)
# effects.remat_allowed_effects.add_type(DebugEffect)
# effects.remat_allowed_effects.add_type(OrderedDebugEffect)
# effects.custom_derivatives_allowed_effects.add_type(DebugEffect)
# effects.custom_derivatives_allowed_effects.add_type(OrderedDebugEffect)

map, unsafe_map = util.safe_map, map

# Sharding visualization
sharding_callbacks = weakref.WeakValueDictionary()  # type: ignore
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
  r, g, b = map(lambda x: int(x, 16), (color[1:3], color[3:5], color[5:7]))
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

# 把sharding画出来
def visualize_sharding(shape: torch.Size, sharding: str, 
                       use_color: bool = True, scale: float = 1.,
                       min_width: int = 9, max_width: int = 80, 
                       color_map: Optional[ColorMap] = None):
                       # shape: Sequence[int], sharding: Sharding, *,
                       # use_color: bool = True, scale: float = 1.,
                       # min_width: int = 9, max_width: int = 80,
                       # color_map: Optional[ColorMap] = None):
  """Visualizes a ``Sharding`` using ``rich``."""
  if not RICH_ENABLED:
    raise ValueError("`visualize_sharding` requires `rich` to be installed.")

  if len(shape) > 2 or len(shape) < 1:
    raise ValueError(
        "`visualize_sharding` only works for shapes with 1 and 2 dimensions.")

  # sharding[sharding.index(']')+1:-1]# sharding.devices_indices_map(tuple(shape))
  slices: dict[tuple[int, ...], set[int]] = {}
  heights: dict[tuple[int, ...], Optional[float]] = {}
  widths: dict[tuple[int, ...], float] = {}

  if len(sharding)>0:
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
      sharding_spac = sharding[sharding.index('['):sharding.index(']')+1]
      if isinstance(sharding[-2], int):
        # eg: '{devices=[2,2]0,1,2,3}' # 13
        device_list = list(sharding[sharding.index(']')+1:-1])
        device_indices_map = [int(i) for i in device_list if i != ',']
        heights = sharding_spac[0]
        widths = sharding_spac[1]
        devices_len = len(device_indices_map)
        for i in range(devices_len+1):
          slices.setdefault((i//widths, i%widths), device_indices_map[i])
      elif sharding[-24:-1] == 'last_tile_dim_replicate':
        device_list = list(sharding[sharding.index(']')+1:-24])
        device_indices_map = [int(i) for i in device_list if i != ',']
        heights = sharding_spac[0]
        widths = sharding_spac[1]
        last_dim_depth = sharding_spac[2]
        devices_len = len(device_indices_map)
        len_after_dim_down = devices_len//last_dim_depth
        for i in range(len_after_dim_down):
          slices.setdefault((i//widths, i%widths), device_indices_map[i:i+last_dim_depth])
      else:
        raise ValueError("sharding is not organized as expected")
  else:
    raise ValueError("sharding has no value")

  # # eg: '{replicated}'
  # if sharding = '{replicated}':
  #   # print it code here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  #   heights = 1
  #   widths = 1
  #   num_devices = xr.global_runtime_device_count()
  #   device_ids = list(range(num_devices))
  #   slices.setdefault((0, 0), device_ids)

  # # eg: '{devices=[2,2]0,1,2,3}' # 13
  # # eg: '{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}' # 15
  # if len(shape) > 2 or len(shape) < 1:
  #   raise ValueError(
  #       "`visualize_sharding` only works for shapes with 1 and 2 dimensions.")

  console = rich.console.Console(width=max_width)
  use_color = use_color and console.color_system is not None
  if use_color and not color_map:
    try:
      import matplotlib as mpl  # pytype: disable=import-error
      color_map = mpl.colormaps["tab20b"]
    except ModuleNotFoundError:
      use_color = False

  base_height = int(10 * scale)
  aspect_ratio = (shape[1] if len(shape) == 2 else 1) / shape[0]
  base_width = int(base_height * aspect_ratio)
  height_to_width_ratio = 2.5

  # eg: '{devices=[2,2]0,1,2,3}' # 13
  # eg: '{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}' # 15

  # slcs is the data we saved on this slice
  # `device_indices_map`: [0, 1, 2, 3]
  # `sharding_spac`: [2, 2]

  # if len(sharding) > 25:
  #   if sharding[-25:-1] == 'last_tile_dim_replicate':
  #     # asd
  #   else:

  # sharding_spac = sharding[sharding.index('['):sharding.index(']')+1]

  # set the device kind to TPU as default since `sharding` here is `str`, TODO(@manfei): get device kind from commands for TPU/GPU/CPU
  device_kind = 'TPU' # next(iter(sharding.device_set)).platform.upper()
  
  # device_list = list(sharding[sharding.index(']')+1:-1])
  # device_indices_map = [int(i) for i in device_list if i != ',']

  # if len(sharding_spac)==2:
  #   # get the table to be printed with 2-dimension sharding_spac
  #   # fake table without getting real data in each devices, TODO: update the table creation code with getting real data from real devices
  #   heights = sharding_spac[0]
  #   widths = sharding_spac[1]
  #   devices_len = len(device_indices_map)
  #   for i in range(devices_len+1):
  #     slices.setdefault((i//widths, i%widths), device_indices_map[i])# .add(int(i))
  # elif len(sharding_spac)==3:
  #   # get the table to be printed with 3-dimension sharding_spac with `last_tile_dim_replicate` labeled
  #   # fake table without getting real data in each devices, TODO: update the table creation code with getting real data from real devices
  #   if device_indices_map[-1]
  #   heights = sharding_spac[0]
  #   widths = sharding_spac[1]
  #   devices_len = len(device_indices_map)
  #   for i in range(devices_len+1):
  #     slices.setdefault((i//widths, i%widths), device_indices_map[i:i+sharding_spac[-1]])# .add(int(i))
  # else
  #   raise ValueError("`visualize_tensor_sharding` only support 2-d or 3-d sharding now, this failure is due to dimension of sharding is not 2 or 3.")

  ## below commneted are code in JAX to get the table in code from devices info with single-device/multi-device/multi-pod
  # for i, (dev) in enumerate(device_indices_map.items()):
  #   assert slcs is not None
  #   slcs = tuple(map(_raise_to_slice, slcs))
  #   chunk_idxs = tuple(map(_slice_to_chunk_idx, shape, slcs))
  #   if slcs is None:
  #     raise NotImplementedError
  #   if len(slcs) == 2:
  #     vert, horiz = slcs
  #     vert_size  = ((vert.stop  - vert.start ) if vert.stop  is not None
  #                   else shape[0])
  #     horiz_size = ((horiz.stop - horiz.start) if horiz.stop is not None
  #                   else shape[1])
  #     chunk_height = vert_size / shape[0] # looks like for multi-hots/multi-pod
  #     chunk_width = horiz_size / shape[1]
  #     heights[chunk_idxs] = chunk_height
  #     widths[chunk_idxs] = chunk_width
  #   else:
  #     # In the 1D case, we set the height to 1.
  #     horiz, = slcs
  #     vert = slice(0, 1, None)
  #     horiz_size = (
  #         (horiz.stop - horiz.start) if horiz.stop is not None else shape[0])
  #     chunk_idxs = (0, *chunk_idxs)
  #     heights[chunk_idxs] = None
  #     widths[chunk_idxs]  = horiz_size / shape[0]
  #   slices.setdefault(chunk_idxs, set()).add(int(dev))
  # num_rows = max([a[0] for a in slices.keys()]) + 1
  # if len(list(slices.keys())[0]) == 1:
  #   num_cols = 1
  # else:
  #   num_cols = max([a[1] for a in slices.keys()]) + 1

  color_iter = make_color_iter(color_map, num_rows, num_cols)
  table = rich.table.Table(show_header=False, show_lines=not use_color,
                           padding=0,
                           highlight=not use_color, pad_edge=False,
                           box=rich.box.SQUARE if not use_color else None)
  for i in range(num_rows):
    col = []
    for j in range(num_cols):
      entry = f"{device_kind} "+",".join([str(s) for s in sorted(slices[i, j])])
      width, maybe_height = widths[i, j], heights[i, j]
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
      padding = tuple(max(x, 0) for x in padding)  # type: ignore
      col.append(
          rich.padding.Padding(
            rich.align.Align(entry, "center", vertical="middle"), padding,
            style=rich.style.Style(bgcolor=color,
              color=text_color)))
    table.add_row(*col)
  console.print(table, end='\n\n')

# # 对value的每个叶节点执行`_inspect`函数
# def inspect_array_sharding(value, *, callback: Callable[[Sharding], None]):
#   def _inspect(val):
#     inspect_sharding_p.bind(val, callback=callback)
#   tree_util.tree_map(_inspect, value) # value是pytree格式存储的， 对每个叶节点执行`_inspect`函数

def visualize_tensor_sharding(ter, **kwargs):
  """Visualizes an array's sharding."""
  import torch_xla
  sharding = torch_xla._XLAC._get_xla_sharding_spec(ter)
  return visualize_sharding(ter.shape, sharding, **kwargs)

  # # arr is the sharding get from torch_xla._XLAC._get_xla_sharding_spec(t)
  # # `_visualize`: 执行visualize_sharding 函数
  # def _visualize(sharding):
  #   # arr is `jaxlib.xla_extension.ArrayImpl`
  #   # sharding could be `PositionalSharding([[{TPU 0} {TPU 2}], [{TPU 1} {TPU 3}]], memory_kind=tpu_hbm)`
  #   return visualize_sharding(tensor.shape, sharding, **kwargs)
  # # `inspect_array_sharding` defined in this file above too
  # # `inspect_array_sharding`： 对arr的每个叶节点执行`_visualize`函数
  # # `_visualize` defined in this file above too
  # inspect_array_sharding(tensor, callback=_visualize)


# what we get from `torch_xla._XLAC._get_xla_sharding_spec`
# ans: string
# eg: '{devices=[2,2]0,1,2,3}'
# eg: '{replicated}'
# eg: '{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}'

# how to print box based on result of `torch_xla._XLAC._get_xla_sharding_spec`

###
# import torch_xla
# sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
# torch_xla._XLAC.visualize_array_sharding(sharding)
# ...boxes...