import contextlib
import io
import itertools
import logging
import sys
import re
import threading
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, TextIO, Tuple, TypedDict, Union
import torch
import torch.distributed._functional_collectives
from torch.library import Library
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
from torch_xla import runtime
import torch_xla.core.xla_env_vars as xenv
import torch_xla.debug.metrics_saver as ms
import torch_xla.utils.utils as xu
import torch_xla.utils.closures as xc
from torch_xla.distributed.spmd.xla_sharding import ShardingSpec
import os
from torch_xla.experimental.deprecation import deprecated
import torch_xla._internal.utils as _utils

_DEVICES = xu.LazyProperty(lambda: torch_xla._XLAC._xla_get_devices())

REDUCE_SUM = 'sum'
REDUCE_MUL = 'mul'
REDUCE_AND = 'and'
REDUCE_OR = 'or'
REDUCE_MIN = 'min'
REDUCE_MAX = 'max'

_DEVICE_CONTEXTS = dict()
_DEVICE_CONTEXTS_LOCK = threading.Lock()

XLA_LIB = Library("xla", "DEF")

from . import xla_model as this_module

xrt_world_size = deprecated(this_module, torch_xla.runtime.world_size,
                            'xrt_world_size() will be removed in release 2.7.')
get_ordinal = deprecated(
    this_module, torch_xla.runtime.global_ordinal,
    'xla_model.get_ordinal() will be removed in release 2.7.')
parse_xla_device = deprecated(
    this_module, _utils.parse_xla_device,
    'xla_model.parse_xla_device() will be removed in release 2.7.')


class DeviceContext(object):

  def __init__(self, device: Union[str, torch.device]):
    self.device = device


def _get_device_context(
    device: Optional[Union[str, torch.device]] = None) -> DeviceContext:
  if device is None:
    device = torch_xla._XLAC._xla_get_default_device()
  else:
    device = str(device)
  with _DEVICE_CONTEXTS_LOCK:
    devctx = _DEVICE_CONTEXTS.get(device, None)
    if devctx is None:
      devctx = DeviceContext(device)
      _DEVICE_CONTEXTS[device] = devctx
    return devctx


def is_xla_tensor(tensor: torch.Tensor) -> bool:
  return tensor.device.type == 'xla'


def get_xla_supported_devices(devkind: Optional[str] = None,
                              max_devices: Optional[int] = None) -> List[str]:
  """Returns a list of supported devices of a given kind.

  Args:
    devkind (string..., optional): If specified, a device type such as `TPU`,
      `CUDA`, `CPU`, or name of custom PJRT device.
    max_devices (int, optional): The maximum number of devices to be returned of
      that kind.

  Returns:
    The list of device strings such as ['xla:0', 'xla:1', ...]
  """
  # TODO(wcromar): Remove `devkind` after 2.3 release cut. We no longer support
  # multiple device types.
  if not devkind:
    devices = torch_xla._XLAC._xla_get_devices()
    return [
        f'xla:{i}'
        for i, _ in enumerate(devices[:max_devices] if max_devices else devices)
    ]
  else:
    warnings.warn("`devkind` argument is deprecated and will be removed in a "
                  "future release.")

  xla_devices = _DEVICES.value
  kind_devices = []
  for i, device in enumerate(xla_devices):
    if re.match(devkind + r':\d+$', device):
      kind_devices.append('xla:{}'.format(i))
  if kind_devices:
    return kind_devices[:max_devices] if max_devices else kind_devices


def get_local_ordinal() -> int:
  """Retrieves the replication local ordinal of the current thread.

  The local ordinals range from 0 to the number of local devices minus 1.

  Returns:
    The replication local ordinal of the current thread.
  """
  return runtime.local_ordinal()


def is_master_ordinal(local: bool = True) -> bool:
  """Checks whether the current process is the master ordinal (0).

  Args:
    local (bool): Whether the local or global master ordinal should be checked.
      In case of multi-host replication, there is only one global master ordinal
      (host 0, device 0), while there are NUM_HOSTS local master ordinals.
      Default: True

  Returns:
    A boolean indicating whether the current process is the master ordinal.
  """
  ordinal = get_local_ordinal() if local else runtime.global_ordinal()
  return ordinal == 0


def master_print(*args: Any,
                 fd: TextIO = sys.stdout,
                 local: bool = False,
                 flush: bool = False):
  if is_master_ordinal(local=local):
    print(*args, file=fd, flush=flush)


def xla_device(n: Optional[int] = None,
               devkind: Optional[str] = None) -> torch.device:
  """Returns a given instance of an XLA device.

  Args:
    n (int, optional): The specific instance (ordinal) to be returned. If
      specified, the specific XLA device instance will be returned. Otherwise
      the first device of `devkind` will be returned.
    devkind (string..., optional): If specified, device type such as `TPU`,
      `CUDA`, `CPU`, or custom PJRT device. Deprecated.

  Returns:
    A `torch.device` with the requested instance.
  """
  # When SPMD is enabled, we always return `xla:0` to the user, and
  # under the hood we use virtual device logic for every xla tensor
  if xu.check_env_flag('XLA_USE_SPMD'):
    device = 'xla:0'
    torch_xla._XLAC._xla_set_default_device(device)
    return torch.device(device)

  return runtime.xla_device(n, devkind)


def _xla_real_device(device: torch.device) -> Any:
  device_str = str(device)
  m = re.match(r'xla:(\d+)$', device_str)
  if not m:
    raise RuntimeError('Invalid device format: {}'.format(device_str))
  return _DEVICES.value[int(m.group(1))]


def xla_real_devices(devices: Optional[List[torch.device]] = None) -> List[str]:
  """Returns the real devices' name.

  Args:
    devices: The list of torch devices such as ['xla:0', 'xla:1'].

  Returns:
    A list of real devices' name such as ['CUDA:0', 'CUDA:1'].
  """
  if not devices:
    devices = get_xla_supported_devices()

  return [_xla_real_device(device) for device in devices]


def xla_device_hw(device: Union[str, torch.device]) -> str:
  """Returns the hardware type of the given device.

  Args:
    device (string or torch.device): The xla device that will be mapped to the
      real device.

  Returns:
    A string representation of the hardware type of the given device.
  """
  real_device = _xla_real_device(device)
  return real_device.split(':')[0]


def xla_device_kind(device: Optional[Union[str, torch.device]] = None) -> str:
  """Returns vendor-dependent string that uniquely identifies the kind of
     device.

  Args:
    device (string or torch.device): The xla device

  Returns:
    A vendor-dependent device kind string.
  """
  if device is None:
    device = torch_xla._XLAC._xla_get_default_device()
  return torch_xla._XLAC._xla_device_kind(str(device))


def xla_replication_devices(
    local_devices: Optional[List[torch.device]] = None) -> List[str]:
  real_devices = xla_real_devices(local_devices)
  device_types = set()
  for device in real_devices:
    xdev = _utils.parse_xla_device(device)
    device_types.add(xdev[0])
  if len(device_types) != 1:
    # No replication if the device set spawns multiple device types.
    raise RuntimeError(
        'Cannot replicate across different device types: devices={}/{}'.format(
            local_devices, real_devices))
  device_type = device_types.pop()
  kind_devices = get_xla_supported_devices()
  if len(kind_devices) != len(local_devices):
    # Replication can only happen among all devices of one kind.
    raise RuntimeError(
        'Cannot replicate if number of devices ({}) is different from {}'.
        format(len(local_devices), len(kind_devices)))
  replication_devices = []
  for device in torch_xla._XLAC._xla_get_all_devices():
    # device is like 'CUDA:0'
    xdev = _utils.parse_xla_device(device)
    if not xdev:
      raise RuntimeError('Invalid device format: {}'.format(device))
    if xdev[0] == device_type:
      replication_devices.append(device)
  sorted_by_ordinal = sorted(
      replication_devices,
      key=lambda device: _utils.parse_xla_device(device)[1])
  return sorted_by_ordinal


def unlazy(tensors: List[torch.Tensor]):
  """Blocks the program until `tensors` are materialized.

  This API is for benchmarking, don't use it in real models.

  Args:
    tensors: List of `torch.Tensor`s to materialize. For each
    Tensor `t` in the list, `t.device` must be an `xla` device.
  """
  torch_xla._XLAC._xla_sync_multi(tensors, devices=[], wait=True)


def set_replication(device: torch.device,
                    devices: Optional[List[torch.device]]):
  device = str(device)
  devctx = _get_device_context(device=device)
  devices = [str(x) for x in devices]
  if devices:
    # sample replication_devices: ['CUDA:0', 'CUDA:1', 'CUDA:2', 'CUDA:3']
    replication_devices = xla_replication_devices(devices)
    torch_xla._XLAC._xla_set_replication_devices(replication_devices)
    devctx.device_index = devices.index(device)
  else:
    torch_xla._XLAC._xla_set_replication_devices([])
    devctx.device_index = 0
  torch_xla._XLAC._set_all_reduce_token(devctx.device, None)
  torch_xla._XLAC._xla_set_default_device(device)


class RateTracker(object):

  def __init__(self, smooth_factor: Optional[float] = None):
    self._smooth_factor = xu.getenv_as(
        'RATE_TRACKER_SMOOTHING', float,
        0.4) if smooth_factor is None else smooth_factor
    self._start_time = time.time()
    self._partial_time = self._start_time
    self._partial_count = 0.0
    self._partial_rate = None
    self._count = 0.0

  def _update(self, now: float, rate: float):
    self._partial_count += self._count
    self._count = 0.0
    self._partial_time = now
    self._partial_rate = rate

  def add(self, count: float):
    self._count += count

  def _smooth(self, current_rate: float) -> float:
    if self._partial_rate is None:
      smoothed_rate = current_rate
    else:
      smoothed_rate = ((1 - self._smooth_factor) * current_rate +
                       self._smooth_factor * self._partial_rate)
    return smoothed_rate

  def rate(self):
    now = time.time()
    delta = now - self._partial_time
    report_rate = 0.0
    if delta > 0:
      report_rate = self._smooth(self._count / delta)
      self._update(now, report_rate)
    return report_rate

  def global_rate(self):
    delta = time.time() - self._start_time
    count = self._partial_count + self._count
    return count / delta if delta > 0 else 0.0


class ToXlaTensorArena(object):

  def __init__(self, convert_fn: Callable[[List[torch.Tensor]],
                                          List[torch.Tensor]],
               select_fn: Callable[[torch.Tensor], bool]):
    self._convert_fn = convert_fn
    self._select_fn = select_fn
    self._tensors = []

  def _add(self, tensor: torch.Tensor):
    self._tensors.append(tensor)

  def _convert(self):
    self._index = 0
    if self._tensors:
      self._converted_tensors = self._convert_fn(self._tensors)
    else:
      self._converted_tensors = []

  def _get_converted_tensor(self) -> torch.Tensor:
    assert self._index < len(self._converted_tensors)
    new_tensor = self._converted_tensors[self._index]
    self._index += 1
    return new_tensor

  def _collect_tensors(self, inputs: Any):

    def collect_fn(value: Any):
      self._add(value)

    xu.for_each_instance(inputs, lambda x: self._select_fn(x), collect_fn)

  def _replace_tensors(self, inputs: Any):

    def convert_fn(value: Any):
      return self._get_converted_tensor()

    return xu.for_each_instance_rewrite(inputs, lambda x: self._select_fn(x),
                                        convert_fn)

  def transform(self, inputs: Any):
    self._tensors = []
    self._collect_tensors(inputs)
    self._convert()
    return self._replace_tensors(inputs)


def check_view_sharing(obj):
  tensors = set()
  aliases = dict()

  def tensor_info(t: torch.Tensor) -> str:
    return '{}{}'.format(t.dtype, list(t.size()))

  def tensor_id(t: torch.Tensor) -> Tuple[int, str]:
    if is_xla_tensor(t):
      return torch_xla._XLAC._xla_get_tensor_id(t), 'xla'
    return id(t), 'torch'

  def alias_id(t: torch.Tensor) -> Tuple[int, str]:
    if is_xla_tensor(t):
      aid = torch_xla._XLAC._xla_get_tensor_view_alias_id(t)
      return None if aid == 0 else aid, 'xla'
    return t.storage().data_ptr(), 'torch'

  def check_object(obj):
    tid = tensor_id(obj)
    if tid not in tensors:
      tensors.add(tid)
      aid = alias_id(obj)
      if aid[0] is not None:
        if aid in aliases:
          oobj = aliases[aid]
          raise RuntimeError(
              'Tensor ID {} ({}) is sharing a view with tensor ID {} ({})'.
              format(tid, tensor_info(obj), tensor_id(oobj), tensor_info(oobj)))
        aliases[aid] = obj

  xu.for_each_instance(obj, lambda x: type(x) == torch.Tensor, check_object)


def _fetch_gradients(optimizer: optim.Optimizer) -> List[torch.Tensor]:
  gradients = []
  for param_group in optimizer.__getstate__()['param_groups']:
    for group, params in param_group.items():
      if group == 'params':
        for p in params:
          if isinstance(p, torch.Tensor) and p.grad is not None:
            gradients.append(p.grad.data)
  return gradients


def _get_all_reduce_token() -> Tuple[Any, DeviceContext]:
  devctx = _get_device_context()
  token = torch_xla._XLAC._get_all_reduce_token(devctx.device)
  return token, devctx


def all_reduce(
    reduce_type: str,
    inputs: Union[torch.Tensor, List[torch.Tensor]],
    scale: float = 1.0,
    groups: Optional[List[List[int]]] = None,
    pin_layout: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
  """Performs an inplace reduce operation on the input tensor(s).

  Args:
    reduce_type (string): One of ``xm.REDUCE_SUM``, ``xm.REDUCE_MUL``,
      ``xm.REDUCE_AND``, ``xm.REDUCE_OR``, ``xm.REDUCE_MIN`` and
      ``xm.REDUCE_MAX``.
    inputs: Either a single `torch.Tensor` or a list of `torch.Tensor` to
      perform the all reduce op to.
    scale (float): A default scaling value to be applied after the reduce.
      Default: 1.0
    groups (list, optional): A list of list, representing the replica groups for
      the `all_reduce()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
        defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
        the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
        all the replicas in it.
    pin_layout (bool, optional): whether to pin the layout for this communication op.
      Layout pining can prevent potential data corruption when each process that
      participate in the communication has slightly different program, but it might
      cause some xla compilation to fail. Unpin the layout when you see error message
      like "HloModule has a mix of layout constrained".

  Returns:
    If a single `torch.Tensor` is passed, the return value is a `torch.Tensor`
    holding the reduced value (across the replicas). If a list/tuple is passed,
    this function performs an inplace all-reduce op on the input tensors, and
    returns the list/tuple itself.
  """
  groups = groups or []

  # No-op if there is only one device
  if runtime.world_size() == 1 and not xu.getenv_as('XLA_ALWAYS_ALLREDUCE',
                                                    bool, False):
    if isinstance(inputs, torch.Tensor):
      return inputs.clone()
    else:
      return inputs

  if isinstance(inputs, torch.Tensor):
    result = None
    if scale == 1.0 and groups == [] and pin_layout:
      # TODO(alanwaketan): Support groups.
      # Only c10d_functional version cc ops are traceable by Dynamo.
      result = torch.ops._c10d_functional.all_reduce(inputs, reduce_type, "")
    else:
      result = torch_xla._XLAC._xla_all_reduce(reduce_type, inputs, scale,
                                               groups, pin_layout)
    results = [result]
  else:
    torch_xla._XLAC._xla_all_reduce_inplace(reduce_type, inputs, scale, groups,
                                            pin_layout)
    results = inputs
  return results[0] if isinstance(inputs, torch.Tensor) else results


def _all_gather_using_all_reduce(
    value: torch.Tensor,
    dim: int = 0,
    groups: Optional[List[List[int]]] = None,
    pin_layout: bool = True) -> Optional[torch.Tensor]:
  """Performs an all-gather operation using all-reduce along a given dimension.

  Args:
    value (torch.Tensor): The input tensor.
    dim (int): The gather dimension.
      Default: 0
    groups (list, optional): A list of list, representing the replica groups for
      the `all_gather()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
        defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
        the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
        all the replicas in it.
    pin_layout (bool, optional): whether to pin the layout for this communication op.
      Layout pining can prevent potential data corruption when each process that
      participate in the communication has slightly different program, but it might
      cause some xla compilation to fail. Unpin the layout when you see error message
      like "HloModule has a mix of layout constrained".

  Returns:
    A tensor which has, in the ``dim`` dimension, all the values from the
    participating replicas.
  """
  if dim < 0:
    dim = value.dim() + dim
  size = value.size(dim)
  padding = [0] * (2 * value.dim())
  ordinal = runtime.global_ordinal()
  if groups is None:
    left, right = ordinal, runtime.world_size() - 1 - ordinal
  else:
    ordinals = dict()
    for g in groups:
      for i, x in enumerate(g):
        ordinals[x] = (i, len(g) - 1 - i)
    left, right = ordinals[ordinal]
  idx = value.dim() - 1 - dim
  padding[2 * idx] = left * size
  padding[2 * idx + 1] = right * size
  return all_reduce(REDUCE_SUM, F.pad(value, padding), groups=groups)


def all_gather(value: torch.Tensor,
               dim: int = 0,
               groups: Optional[List[List[int]]] = None,
               output: Optional[torch.Tensor] = None,
               pin_layout: bool = True) -> torch.Tensor:
  """Performs an all-gather operation along a given dimension.

  Args:
    value (torch.Tensor): The input tensor.
    dim (int): The gather dimension.
      Default: 0
    groups (list, optional): A list of list, representing the replica groups for
      the `all_gather()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
        defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
        the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
        all the replicas in it.
    output (torch.Tensor): Optional output tensor.
    pin_layout (bool, optional): whether to pin the layout for this communication op.
      Layout pining can prevent potential data corruption when each process that
      participate in the communication has slightly different program, but it might
      cause some xla compilation to fail. Unpin the layout when you see error message
      like "HloModule has a mix of layout constrained".

  Returns:
    A tensor which has, in the ``dim`` dimension, all the values from the
    participating replicas.
  """
  # _all_gather_using_all_reduce does not support list of tensors as input
  if pin_layout and output == None and isinstance(value, torch.Tensor):
    # There is not an easy way to pin the all_gather layout, so use all_reduce
    # based all_gather for this purpose.
    return _all_gather_using_all_reduce(
        value, dim=dim, groups=groups, pin_layout=True)

  if dim < 0:
    dim = value.dim() + dim
  if groups:
    shard_count = len(groups[0])
    assert all(len(group) == shard_count for group in groups), \
      "Replica groups must have the same number of replicas/shards."
  else:
    # All replicas belong to a single group
    shard_count = runtime.world_size()

  token, devctx = _get_all_reduce_token()

  if isinstance(value, torch.Tensor):
    if output != None:
      # Call the out of place version of the all_gather
      new_token = torch_xla._XLAC._xla_all_gather_out(output, value, token, dim,
                                                      shard_count, groups or [],
                                                      pin_layout)
      torch_xla._XLAC._set_all_reduce_token(devctx.device, new_token)
      return output

    result = torch_xla._XLAC._xla_all_gather(value, dim, shard_count, groups or
                                             [], pin_layout)
    return result

  # Now the input should be a list of Tensors.
  elif isinstance(value, list) and all(
      isinstance(v, torch.Tensor) for v in value):
    if pin_layout:
      raise RuntimeError(
          "For xm.all_gather with list of tensors input, pin_layout=True is not yet supported."
      )
    if output != None:
      if not isinstance(output, list) or any(
          not isinstance(v, torch.Tensor) for v in output):
        raise TypeError(
            f"`output` needs to be a list of Tensors, but given {type(output)}."
        )
      if len(output) != len(value):
        raise ValueError("`output` length doesn't match `input` length: "
                         f"{len(output)} vs {len(input)}.")
      # Call the out of place version of the reduce_scatter
      new_token = torch_xla._XLAC._xla_all_gather_coalesced_out(
          output, value, token, dim, shard_count, groups or [], pin_layout)
      torch_xla._XLAC._set_all_reduce_token(devctx.device, new_token)
      return output

    result = torch_xla._XLAC._xla_all_gather_coalesced(value, token, dim,
                                                       shard_count, groups or
                                                       [], pin_layout)
    torch_xla._XLAC._set_all_reduce_token(devctx.device, result[-1])
    return result[:-1]
  else:
    raise TypeError("`value` needs to be a Tensor or a list of Tensors, but "
                    f"given {type(value)}.")


class CoalescingBuckets(object):

  def __init__(
      self,
      func: Callable[[
          Union[torch.Tensor,
                List[torch.Tensor]], Optional[Union[torch.Tensor,
                                                    List[torch.Tensor]]]
      ], Union[torch.Tensor, List[torch.Tensor]]],
      input_list: Any,
      output_list: Optional[Any] = None,
      bucket_cap_mb: int = 160):
    if not isinstance(input_list, list) or any(
        not isinstance(v, torch.Tensor) for v in input_list):
      raise TypeError(
          f"`input_list` needs to be a list of Tensors, but given {type(input_list)}."
      )
    if output_list != None:
      if not isinstance(output_list, list) or any(
          not isinstance(v, torch.Tensor) for v in output_list):
        raise TypeError(
            f"`output_list` needs to be a list of Tensors, but given {type(output_list)}."
        )
      if len(output_list) != len(input_list):
        raise ValueError(
            "`output_list` length doesn't match `input_list` length: "
            f"{len(output_list)} vs {len(input_list)}.")
    self._func = func
    self._input_list = input_list
    self._output_list = output_list
    self._total = 0
    self._tensor_bucket = []
    self._output_bucket = [] if output_list else None
    self._bucket_cap = bucket_cap_mb * 1024 * 1024
    self._out_tensors = []

  def flush(self):
    if len(self._tensor_bucket) == 1:
      # Use non-coalesced CCOp if its just one tensor
      output = self._output_bucket[0] if self._output_bucket else None
      self._out_tensors.append(self._func(self._tensor_bucket[0], output))
    elif len(self._tensor_bucket):
      self._out_tensors.extend(
          self._func(self._tensor_bucket, self._output_bucket))
    self._total = 0
    self._tensor_bucket = []
    self._output_bucket = [] if self._output_list else None

  def add(self, tensor: torch.Tensor, idx: int):
    self._total += tensor.numel() * tensor.element_size()
    self._tensor_bucket.append(tensor)
    if self._output_list != None:
      self._output_bucket.append(self._output_list[idx])

  def __call__(self) -> Union[torch.Tensor, List[torch.Tensor]]:
    for idx, tensor in enumerate(self._input_list):
      tensor_bytes = tensor.numel() * tensor.element_size()

      # Aim for target bucket_cap_mb: flush new tensor with bucket if bucket content
      # is small (1/2 cap) but don't combine if combined total is over 2x cap
      total_new = self._total + tensor_bytes
      if tensor_bytes > self._bucket_cap and self._total < 0.5 * self._bucket_cap and total_new <= 2 * self._bucket_cap:
        self.add(tensor, idx)
        self.flush()
      else:
        # Bucketize till the total spills over
        if total_new > self._bucket_cap:
          self.flush()
        self.add(tensor, idx)

    # Flush the last remaining bucket
    self.flush()

    assert len(self._out_tensors) == len(self._input_list)

    return self._out_tensors


def all_gather_bucketized(
    input_list: List[torch.Tensor],
    dim: int = 0,
    groups: Optional[List[List[int]]] = None,
    output: Optional[torch.Tensor] = None,
    pin_layout: bool = False,
    bucket_cap_mb=160) -> Union[torch.Tensor, List[torch.Tensor]]:
  """Performs an all-gather operation along a given dimension, with bucketization.

  Args:
    See all_gather for the args: dim, groups, output, pin_layout
    input_list: List of input tensors
    bucket_cap_mb: Number of MegaBytes of the tensor bucket to fill before doing all-gather.

  Returns:
    A list of tensors each of which has, in the ``dim`` dimension, all the values from the
    participating replicas.
  """
  # sanity checks
  if pin_layout:
    raise RuntimeError(
        "For xm.all_gather_bucketized, pin_layout=True is not yet supported.")

  def _all_gather_coalesced(_input_list, _output_list=None):
    return all_gather(
        value=_input_list,
        dim=dim,
        groups=groups,
        output=_output_list,
        pin_layout=pin_layout)

  buckets = CoalescingBuckets(
      _all_gather_coalesced, input_list, output, bucket_cap_mb=bucket_cap_mb)
  return buckets()


def all_to_all(value: torch.Tensor,
               split_dimension: int,
               concat_dimension: int,
               split_count: int,
               groups: Optional[List[List[int]]] = None,
               pin_layout: bool = True) -> torch.Tensor:
  """Performs an XLA `AllToAll()` operation on the input tensor.

  See: https://www.tensorflow.org/xla/operation_semantics#alltoall

  Args:
    value (torch.Tensor): The input tensor.
    split_dimension (int): The dimension upon which the split should happen.
    concat_dimension (int): The dimension upon which the concat should happen.
    split_count (int): The split count.
    groups (list, optional): A list of list, representing the replica groups for
      the `all_reduce()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
        defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
        the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
        all the replicas in it.
    pin_layout (bool, optional): whether to pin the layout for this communication op.
      Layout pining can prevent potential data corruption when each process that
      participate in the communication has slightly different program, but it might
      cause some xla compilation to fail. Unpin the layout when you see error message
      like "HloModule has a mix of layout constrained".

  Returns:
    The result `torch.Tensor` of the `all_to_all()` operation.
  """
  token, devctx = _get_all_reduce_token()
  result = torch_xla._XLAC._xla_all_to_all(value, token, split_dimension,
                                           concat_dimension, split_count,
                                           groups or [], pin_layout)
  torch_xla._XLAC._set_all_reduce_token(devctx.device, result[1])
  return result[0]


def collective_permute(value: torch.Tensor,
                       pairs: List[List[int]]) -> torch.Tensor:
  """Performs a XLA `CollectivePermute()` operation on the input tensor.

  WARNING: This function is not very reliable, may produce wrong results under
           certain inputs. Use it at your own risk.

  See: https://www.tensorflow.org/xla/operation_semantics#collectivepermute

  Args:
    value (torch.Tensor): The input tensor.
    pairs (list): A list of (source_replica_id, target_replica_id) pairs,
      representing the sender and receiver for the `collective_permute()`
      operation. Example: `[[0, 1], [1, 2], [2, 0]]` defines three pairs. The
        tensor will be sent from replica 0 to replica 1, replica 1 to replica 2,
        and replica 2 to replica 0.

  Returns:
    The result `torch.Tensor` of the `collective_permute()` operation.
  """
  token, devctx = _get_all_reduce_token()
  result = torch_xla._XLAC._xla_collective_permute(value, token, pairs)
  torch_xla._XLAC._set_all_reduce_token(devctx.device, result[1])
  return result[0]


def collective_broadcast(tensors: List[torch.Tensor],
                         root_ordinal: int = 0,
                         groups: Optional[List[int]] = None,
                         pin_layout: bool = True) -> None:
  """Broadcast values of `tensors` from root replica to other replicas in-place.

  Args:
    tensors (list): List of `torch.Tensor`s to broadcast.
    root_ordinal (int): Ordinal of replica with values to broadcast.
    groups (list, optional): A list of list, representing the replica groups for
      the `all_reduce()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
        defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
        the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
        all the replicas in it.
    pin_layout (bool, optional): whether to pin the layout for this communication op.
      Layout pining can prevent potential data corruption when each process that
      participate in the communication has slightly different program, but it might
      cause some xla compilation to fail. Unpin the layout when you see error message
      like "HloModule has a mix of layout constrained".
  """
  with torch.no_grad():
    # We must produce the exact same graph in each replica to prevent hanging,
    # so each replica must have the same multiply op with the same parameters.
    for tensor in tensors:
      scale = torch.tensor(
          1 if runtime.global_ordinal() == root_ordinal else 0,
          dtype=tensor.dtype)
      # Transfer scale tensor as device data instead of constant 1 or 0.
      xscale = send_cpu_data_to_device(scale, tensor.device)
      tensor.mul_(xscale[0])

  all_reduce(REDUCE_SUM, tensors, groups=groups, pin_layout=pin_layout)


def send(value: torch.Tensor, channel_id: int) -> torch.Tensor:
  """Performs a XLA `Send()` operation on the input tensor.

  See: https://www.tensorflow.org/xla/operation_semantics#send

  Args:
    value (torch.Tensor): The input tensor.
    channel_id (int64): opaque id identifying the destination of the send op.
  """
  token, devctx = _get_all_reduce_token()
  # The input will be returned as result.
  input_as_result, new_token = torch_xla._XLAC._xla_send(
      value, token, channel_id)
  torch_xla._XLAC._set_all_reduce_token(devctx.device, new_token)
  return input_as_result


def recv(output: torch.Tensor, channel_id: int) -> torch.Tensor:
  """Performs a XLA `Recv()` operation on the input tensor.

  See: https://www.tensorflow.org/xla/operation_semantics#recv

  Args:
    output (torch.Tensor): The output tensor.
    channel_id (int64): opaque id identifying the source of the recv op.
  """
  token, devctx = _get_all_reduce_token()
  result, new_token = torch_xla._XLAC._xla_recv(output, token, channel_id)
  torch_xla._XLAC._set_all_reduce_token(devctx.device, new_token)
  return result


def reduce_scatter(reduce_type: str,
                   input: Union[torch.Tensor, List[torch.Tensor]],
                   scale: float,
                   scatter_dim: int,
                   shard_count: int,
                   groups: Optional[List[List[int]]] = None,
                   output: Optional[Union[torch.Tensor,
                                          List[torch.Tensor]]] = None,
                   pin_layout: bool = True) -> torch.Tensor:
  """Performs a XLA `ReduceScatter()` operation on the input tensor.

  See: https://www.tensorflow.org/xla/operation_semantics#reducescatter

  Args:
    reduce_type (string): One of ``xm.REDUCE_SUM``, ``xm.REDUCE_MUL``,
      ``xm.REDUCE_AND``, ``xm.REDUCE_OR``, ``xm.REDUCE_MIN`` and
      ``xm.REDUCE_MAX``.
    input: (torch.Tensor or a list of torch.Tensor): The input. If it's a list, then
      it will also be the output.
    scale (float): A default scaling value to be applied after the reduce.
    scatter_dim (int): Dimension number to which apply scatter operation.
    shard_count (int): The number of ways to split up the scatter_dim in.
    groups (list): A list of list, representing the replica groups for
      the `reduce_scatter()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
        defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
        the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
        all the replicas in it.
    output: Optional output tensor if `input` is a torch.Tensor, or a list of
      torch.Tensor if `input` is a list of torch.Tensor.
    pin_layout (bool, optional): whether to pin the layout for this communication op.
      Layout pining can prevent potential data corruption when each process that
      participate in the communication has slightly different program, but it might
      cause some xla compilation to fail. Unpin the layout when you see error message
      like "HloModule has a mix of layout constrained".

  Returns:
    A `torch.Tensor` with all the values reduced across replicas. Each process
    gets a shard split along the `scatter_dim`. All other dimensions are
    the same as the input.
  """
  token, devctx = _get_all_reduce_token()

  if isinstance(input, torch.Tensor):
    if output != None:
      # Call the out of place version of the reduce_scatter
      new_token = torch_xla._XLAC._xla_reduce_scatter_out(
          reduce_type, output, input, token, scale, scatter_dim, shard_count,
          groups or [], pin_layout)
      torch_xla._XLAC._set_all_reduce_token(devctx.device, new_token)
      return output

    result = torch_xla._XLAC._xla_reduce_scatter(reduce_type, input, token,
                                                 scale, scatter_dim,
                                                 shard_count, groups or [],
                                                 pin_layout)
    torch_xla._XLAC._set_all_reduce_token(devctx.device, result[1])
    return result[0]

  # Now the input should be a list of Tensors.
  elif isinstance(input, list) and all(
      isinstance(v, torch.Tensor) for v in input):
    if output != None:
      if not isinstance(output, list) or any(
          not isinstance(v, torch.Tensor) for v in output):
        raise TypeError(
            f"`output` needs to be a list of Tensors, but given {type(output)}."
        )
      if len(output) != len(input):
        raise ValueError("`output` length doesn't match `input` length: "
                         f"{len(output)} vs {len(input)}.")
      # Call the out of place version of the reduce_scatter
      new_token = torch_xla._XLAC._xla_reduce_scatter_coalesced_out(
          reduce_type, output, input, token, scale, scatter_dim, shard_count,
          groups or [], pin_layout)
      torch_xla._XLAC._set_all_reduce_token(devctx.device, new_token)
      return output

    result = torch_xla._XLAC._xla_reduce_scatter_coalesced(
        reduce_type, input, token, scale, scatter_dim, shard_count, groups or
        [], pin_layout)
    torch_xla._XLAC._set_all_reduce_token(devctx.device, result[-1])
    return result[:-1]
  else:
    raise TypeError("`input` needs to be a Tensor or a list of Tensors, but "
                    f"given {type(input)}.")


def reduce_scatter_bucketized(reduce_type: str,
                              input_list: Union[torch.Tensor,
                                                List[torch.Tensor]],
                              scale: float,
                              scatter_dim: int,
                              shard_count: int,
                              groups: Optional[List[List[int]]] = None,
                              output: Optional[Union[
                                  torch.Tensor, List[torch.Tensor]]] = None,
                              pin_layout: bool = False,
                              bucket_cap_mb: int = 160) -> CoalescingBuckets:
  """Performs a XLA `ReduceScatter()` operation on a list of tensors (bucketized).

  See: https://www.tensorflow.org/xla/operation_semantics#reducescatter

  Args:
    see reduce_scatter for reduce_type, scale, scatter_dim, shard_count, groups, pin_layout
    input_list: List of input tensors
    output: Optional list of output torch.Tensor
    bucket_cap_mb: Number of MegaBytes of the tensor bucket to fill before doing reduce-scatter.

  Returns:
    A list of `torch.Tensors` with all the values reduced across replicas. Each process
    gets a shard split along the `scatter_dim`. All other dimensions are
    the same as the input.
  """

  def _reduce_scatter_coalesced(
      _input_list: Union[torch.Tensor, List[torch.Tensor]],
      _output_list: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
  ) -> Union[torch.Tensor, List[torch.Tensor]]:
    return reduce_scatter(
        reduce_type=reduce_type,
        input=_input_list,
        scale=scale,
        scatter_dim=scatter_dim,
        shard_count=shard_count,
        groups=groups,
        output=_output_list,
        pin_layout=pin_layout)

  buckets = CoalescingBuckets(
      _reduce_scatter_coalesced,
      input_list,
      output,
      bucket_cap_mb=bucket_cap_mb)
  return buckets()


def add_step_closure(closure: Callable[..., Any],
                     args: Tuple[Any, ...] = (),
                     run_async: bool = False):
  """Adds a closure to the list of the ones to be run at the end of the step.

  Many times during model training there is the need to print/report (print to
  console, post to tensorboard, etc...) information which require the content of
  intermediary tensors to be inspected.
  Inspecting different tensors content in different points of the model code
  requires many executions and typically causes performance issues.
  Adding a step closure will ensure that it will be run after the barrier, when
  all the live tensors will be already materialized to device data.
  Live tensors which will include the ones captured by the closure arguments.
  So using `add_step_closure()` will ensure a single execution will be
  performed, even when multiple closures are queued, requiring multiple tensors
  to be inspected.
  Step closures will be run sequentially in the order they have been queued.
  Note that even though using this API the execution will be optimized, it is
  advised to throttle the printing/reporting events once every N steps.

  Args:
    closure (callable): The function to be called.
    args (tuple): The arguments to be passed to the closure.
    run_async: If True, run the closure asynchronously.
  """
  devctx = _get_device_context()
  closures_type = 'async_step_closures' if run_async else 'step_closures'
  step_closures = getattr(devctx, closures_type, None)
  if step_closures is None:
    step_closures = []
    setattr(devctx, closures_type, step_closures)
  step_closures.append(lambda a=args: closure(*a))


def _run_step_closures() -> DeviceContext:
  devctx = _get_device_context()
  async_step_closures = getattr(devctx, 'async_step_closures', None)
  if async_step_closures is not None:
    devctx.async_step_closures = []
    async_closure_handler = getattr(devctx, 'async_closure_handler', None)
    if async_closure_handler is None:
      async_closure_handler = xc.AsyncClosureHandler()
      devctx.async_closure_handler = async_closure_handler
    async_closure_handler.run_all(async_step_closures)

  step_closures = getattr(devctx, 'step_closures', None)
  if step_closures is not None:
    devctx.step_closures = []
    for closure in step_closures:
      closure()
  return devctx


def mark_step(wait: bool = False, reset_scope: bool = True):
  if xu.getenv_as('XLA_EMIT_STEPLOG', bool, False):
    print(
        'torch_xla.core.xla_model::mark_step\n',
        end='',
        file=sys.stderr,
        flush=True)
  torch_xla._XLAC._xla_step_marker(
      torch_xla._XLAC._xla_get_default_device(), [],
      wait=xu.getenv_as('XLA_SYNC_WAIT', bool, wait),
      reset_scope=reset_scope)
  # Only emit metrics from the first local device index, to avoid emitting the
  # same values from different threads.
  if is_master_ordinal():
    ms.save_metrics()
  devctx = _run_step_closures()
  torch_xla._XLAC._set_all_reduce_token(devctx.device, None)


# TODO(lsy323): When `tensors` is empty, the some intermediate tensors will also be
# dump as outputs. Need further investigation.
def get_stablehlo(tensors: Optional[List[torch.Tensor]] = None) -> str:
  """Get StableHLO for the computation graph in string format.

  If `tensors` is not empty, the graph with `tensors` as outputs will be dump.
  If `tensors` is empty, the whole computation graph will be dump.

  For inference graph, it is recommended to pass the model outputs to `tensors`.
  For training graph, it is not straightforward to identify the "outputs". Using empty `tensors` is recommended.

  To enable source line info in StableHLO, please set env var XLA_HLO_DEBUG=1.

  Args:
    tensors (list[torch.Tensor], optional): Tensors that represent the output/root of the StableHLO graph.

  Returns:
    StableHLO Module in string format.
  """
  if tensors is None:
    tensors = []
  return torch_xla._XLAC._get_stablehlo(
      tensors, torch_xla._XLAC._xla_get_default_device(), [],
      False).decode('utf-8')


# TODO(lsy323): When `tensors` is empty, the some intermediate tensors will also be
# dump as outputs. Need further investigation.
def get_stablehlo_bytecode(tensors: Optional[torch.Tensor] = None) -> bytes:
  """Get StableHLO for the computation graph in bytecode format.

  If `tensors` is not empty, the graph with `tensors` as outputs will be dump.
  If `tensors` is empty, the whole computation graph will be dump.

  For inference graph, it is recommended to pass the model outputs to `tensors`.
  For training graph, it is not straightforward to identify the "outputs". Using empty `tensors` is recommended.

  Args:
    tensors (list[torch.Tensor], optional): Tensors that represent the output/root of the StableHLO graph.

  Returns:
    StableHLO Module in bytecode format.
  """
  if tensors is None:
    tensors = []
  return torch_xla._XLAC._get_stablehlo(
      tensors, torch_xla._XLAC._xla_get_default_device(), [], True)


def wait_device_ops(devices: List[str] = []):
  """Waits for all the async operations on the given devices to complete.

  Args:
    devices (string..., optional): The devices whose async ops need to be waited
      for. If empty, all the local devices will be waited for.
  """
  torch_xla._XLAC._xla_wait_device_ops(devices=devices)


def all_reduce_bucketized_gradients(gradients: List[torch.Tensor],
                                    scale: float,
                                    groups: Optional[List[List[int]]],
                                    pin_layout: bool,
                                    bucket_cap_mb: int = 0):
  total = 0
  tensor_bucket = []
  bucket_cap = bucket_cap_mb * 1024 * 1024

  for grad in gradients:
    grad_bytes = grad.numel() * grad.element_size()

    # Bucketize till the total spills over
    total += grad_bytes
    if total > bucket_cap and len(tensor_bucket) > 0:
      all_reduce(
          REDUCE_SUM,
          tensor_bucket,
          scale=scale,
          groups=groups,
          pin_layout=pin_layout)
      total = grad_bytes
      tensor_bucket = []
    tensor_bucket.append(grad)

  # Flush the last remaining bucket
  if len(tensor_bucket):
    all_reduce(
        REDUCE_SUM,
        tensor_bucket,
        scale=scale,
        groups=groups,
        pin_layout=pin_layout)


def reduce_gradients(optimizer: optim.Optimizer,
                     groups: Optional[List[List[int]]] = None,
                     pin_layout: bool = True):
  """Reduces all the gradients handled by an optimizer.

  Args:
    optimizer (:class:`torch.Optimizer`): The `torch.Optimizer` instance
      containing the gradients to be reduced.
    groups (list, optional): A list of list, representing the replica groups for
      the `all_reduce()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
        defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
        the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
        all the replicas in it.
    pin_layout (bool, optional): whether to pin the layout when reducing gradients.
      See `xm.all_reduce` for details.
  """
  count = runtime.world_size()
  if count > 1:
    gradients = _fetch_gradients(optimizer)
    bucket_cap_mb = int(os.getenv('ALLREDUCE_GRADIENTS_BUCKET_SIZE_MB', 0))
    # Reverse the gradients list so that we start allreduce from the last layer
    # onwards. This allows allreduce to trigger as soon as the bucket fills up and
    # overlap with backward pass.
    if bucket_cap_mb > 0:
      gradients = reversed(gradients)
      all_reduce_bucketized_gradients(
          gradients,
          scale=1.0 / count,
          groups=groups,
          pin_layout=pin_layout,
          bucket_cap_mb=bucket_cap_mb)
    else:
      all_reduce(
          REDUCE_SUM,
          gradients,
          scale=1.0 / count,
          groups=groups,
          pin_layout=pin_layout)


def optimizer_step(optimizer: optim.Optimizer,
                   barrier: bool = False,
                   optimizer_args: Dict = {},
                   groups: Optional[List[List[int]]] = None,
                   pin_layout: bool = True):
  """Run the provided optimizer step and sync gradidents across all devices.

  Args:
    optimizer (:class:`torch.Optimizer`): The `torch.Optimizer` instance whose
      `step()` function needs to be called. The `step()` function will be called
      with the `optimizer_args` named arguments.
    barrier (bool, optional): Whether the XLA tensor barrier should be issued in
      this API. If using the PyTorch XLA `ParallelLoader` or `DataParallel`
      support, this is not necessary as the barrier will be issued by the XLA
      data loader iterator `next()` call.
      Default: False
    optimizer_args (dict, optional): Named arguments dictionary for the
      `optimizer.step()` call.
    groups (list, optional): A list of list, representing the replica groups for
      the `all_reduce()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
        defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
        the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
        all the replicas in it.
    pin_layout (bool, optional): whether to pin the layout when reducing gradients.
      See `xm.all_reduce` for details.

  Returns:
    The same value returned by the `optimizer.step()` call.

  Example:

    >>> import torch_xla.core.xla_model as xm
    >>> xm.optimizer_step(self.optimizer)
  """
  reduce_gradients(optimizer, groups=groups, pin_layout=pin_layout)
  loss = optimizer.step(**optimizer_args)
  if barrier:
    mark_step()
  return loss


def save(data: Any,
         file_or_path: Union[str, TextIO],
         master_only: bool = True,
         global_master: bool = False):
  """Saves the input data into a file.

  The saved data is transferred to PyTorch CPU device before being saved, so a
  following `torch.load()` will load CPU data.
  Care must be taken when working with views. Instead of saving views it's
  recommended that you recreate them after the tensors have been loaded and
  moved to their destination device(s).

  Args:
    data: The input data to be saved. Any nested combination of Python objects
      (list, tuples, sets, dicts, ...).
    file_or_path: The destination for the data saving operation. Either a file
      path or a Python file object. If `master_only` is ``False`` the path or
      file objects must point to different destinations as otherwise all the
      writes from the same host will override each other.
    master_only (bool, optional): Whether only the master device should save the
      data. If False, the `file_or_path` argument should be a different file or
      path for each of the ordinals taking part to the replication, otherwise
      all the replicas on the same host will be writing to the same location.
      Default: True
    global_master (bool, optional): When ``master_only`` is ``True`` this flag
      controls whether every host's master (if ``global_master`` is ``False``)
      saves the content, or only the global master (ordinal 0).
      Default: False

  Example:

    >>> import torch_xla.core.xla_model as xm
    >>> xm.wait_device_ops() # wait for all pending operations to finish.
    >>> xm.save(obj_to_save, path_to_save)
    >>> xm.rendezvous('torch_xla.core.xla_model.save') # multi process context only
  """
  should_write_data = not master_only or is_master_ordinal(
      local=not global_master)

  cpu_data = _maybe_convert_to_cpu(data, convert=should_write_data)
  if should_write_data:
    torch.save(cpu_data, file_or_path)


def _maybe_convert_to_cpu(data: Any, convert: bool = True) -> ToXlaTensorArena:

  def convert_fn(tensors):
    torch_xla._XLAC._xla_sync_multi(
        tensors, devices=[], wait=True, sync_xla_data=False)
    if not convert:
      return tensors
    return torch_xla._XLAC._xla_get_cpu_tensors(tensors)

  def select_fn(v):
    return type(v) == torch.Tensor and is_xla_tensor(v)

  return ToXlaTensorArena(convert_fn, select_fn).transform(data)


def send_cpu_data_to_device(
    datas: Any,
    device: Union[str, torch.device],
    input_sharding: Optional[ShardingSpec] = None) -> ToXlaTensorArena:

  def convert_fn(tensors):
    devices = [str(device)] * len(tensors)
    shardings = None
    if input_sharding:
      shardings = [input_sharding.xla_spec(t) for t in tensors]
    if input_sharding and input_sharding.minibatch:
      # when minibatch is configured we must make sure batch dimension of
      # the tensor is divisible by the local runtime device count.
      for tensor, sharding in zip(tensors, shardings):
        # assume batch dimension is 0
        local_runtime_device_count = torch_xla.runtime.addressable_runtime_device_count(
        )
        if sharding and tensor.dim() > 0 and (tensor.size()[0] %
                                              local_runtime_device_count) != 0:
          raise RuntimeError(
              "When minibatch is configured, the per-host batch size must be divisible "
              + "by local runtime device count. Per host input data shape " +
              f"= {tensor.size()}, local_runtime_device_count = {local_runtime_device_count}"
          )

    xtensors = torch_xla._XLAC._xla_tensors_from_aten(tensors, devices,
                                                      shardings)
    return xtensors

  def select_fn(v):
    return type(v) == torch.Tensor and v.device.type == 'cpu'

  if type(datas) is torch.Tensor:
    datas = [datas]
  return ToXlaTensorArena(convert_fn, select_fn).transform(datas)


def xla_rendezvous(payload: bytes = b'',
                   ordinals: Optional[List[int]] = None,
                   tag: Optional[str] = None) -> List[bytes]:
  """Share `payload` with all replicas in `ordinals`.

  `tag` is ignored except for logging.

  Uses XLA collective communication to communicate between replicas, so this
  will sync the graph (`xm.mark_step`).

  Args:
    tag: Name of this rendezvous operation.
    payload: Payload to share with other replicas.
    ordinals: List of replicas participating in rendezvous.
  Returns:
    List of bytes from other replicas.
  """
  if ordinals and len(ordinals) != runtime.global_device_count():
    raise ValueError('Only global rendezvous is supported')

  if not isinstance(payload, bytes):
    raise TypeError('`payload` must be bytes, not {}'.format(type(payload)))

  # Finish all execution of previous graphs to avoid recompilation
  mark_step()

  device = xla_device()

  data = torch.tensor(list(payload), dtype=torch.uint8)
  size = torch.tensor([data.shape[0]], dtype=torch.int, device=device)

  if tag:
    logging.info(f"Joining rendezvous '{tag}'...")

  sizes = all_gather(size)

  max_size = torch.max(sizes)
  mark_step()

  # If all payloads are empty, return immediately to avoid more CPU transfers
  if max_size.item() < 1:
    return [b'' for _ in range(sizes.size()[0])]

  padded_data = torch.nn.functional.pad(data, (
      0,
      max_size.item() - size.item(),
  )).to(xla_device())
  raw_data = all_gather(padded_data)
  data_list = torch.split(raw_data, max_size)

  payloads = [d[:sz] for d, sz in zip(data_list, sizes.cpu())]
  mark_step()

  return [bytes(p.cpu().tolist()) for p in payloads]


def rendezvous(tag: str,
               payload: bytes = b'',
               replicas: List[int] = []) -> List[bytes]:
  """Waits for all the mesh clients to reach the named rendezvous.

  Note: PJRT does not support the XRT mesh server, so this is effectively an
  alias to `xla_rendezvous`.

  Args:
    tag (string): The name of the rendezvous to join.
    payload (bytes, optional): The payload to be sent to the rendezvous.
    replicas (list, int): The replica ordinals taking part of the rendezvous.
      Empty means all replicas in the mesh.
      Default: []

  Returns:
    The payloads exchanged by all the other cores, with the payload of core
    ordinal `i` at position `i` in the returned tuple.

  Example:

    >>> import torch_xla.core.xla_model as xm
    >>> xm.rendezvous('example')
  """
  return xla_rendezvous(payload, replicas or None, tag=tag)


def do_on_ordinals(
    target: Callable[..., Any],
    data: Union[Tuple, Any] = (),
    ordinals: Union[List[int], Set[int], int] = (0,)
) -> Optional[Any]:
  """Runs a function only on a given set of ordinals.

  Args:
    target (callable): The function to be run on `ordinals`.
    data: Any input data for the `target` function which contains tensors. All
      the XLA tensors used by the `target` function must be passed in this
      argument. Every other data used by the function can be captured by the
      Python interpreter as usual.
      Default: ()
    ordinals (list, int): The list/set of ordinals where the `target` function
      should run.
      Default: (0,)

  Returns:
    In the ordinals that ran the `target` function, the function return value,
    otherwise `None`.
  """
  running = runtime.global_ordinal() in ordinals
  cpu_data = _maybe_convert_to_cpu(data, convert=running)
  if running:
    result = target(*cpu_data)
  else:
    result = None
  rendezvous('torch_xla.core.xla_model.do_on_ordinals')
  return result


def mesh_reduce(tag: str, data,
                reduce_fn: Callable[..., Any]) -> Union[Any, ToXlaTensorArena]:
  """Performs an out-of-graph client mesh reduction.

  Args:
    tag (string): The name of the rendezvous to join.
    data: The data to be reduced. The `reduce_fn` callable will receive a list
      with the copies of the same data coming from all the mesh client processes
      (one per core).
    reduce_fn (callable): A function which receives a list of `data`-like
      objects and returns the reduced result.

  Returns:
    The reduced value.

  Example:

    >>> import torch_xla.core.xla_model as xm
    >>> import numpy as np
    >>> accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
  """
  cpu_data = _maybe_convert_to_cpu(data)
  bio = io.BytesIO()
  torch.save(cpu_data, bio)
  xdata = rendezvous(tag, bio.getvalue())
  xldata = []
  for xd in xdata:
    xbio = io.BytesIO(xd)
    xldata.append(torch.load(xbio))
  return reduce_fn(xldata) if xldata else cpu_data


def set_rng_state(seed: int, device: Optional[str] = None):
  """Sets the random number generator state.

  Args:
    seed (integer): The state to be set.
    device (string, optional): The device where the RNG state needs to be set.
      If missing the default device seed will be set.
  """
  if device is None:
    device = torch_xla._XLAC._xla_get_default_device()
  torch_xla._XLAC._xla_set_rng_seed(seed, str(device) if device else '')


def get_rng_state(device: Optional[str] = None) -> int:
  """Gets the current running random number generator state.

  Args:
    device (string, optional): The device whose RNG state needs to be retrieved.
      If missing the default device seed will be set.

  Returns:
    The RNG state, as integer.
  """
  if device is None:
    device = torch_xla._XLAC._xla_get_default_device()
  return torch_xla._XLAC._xla_get_rng_seed(str(device) if device else '')


@contextlib.contextmanager
def fork_rng(device: Optional[str] = None, enabled: bool = True):
  """
  Forks the RNG, so that when you return, the RNG is reset to the state that it was previously in.
  Args:
    device (string, optional): The device where the RNG state needs to be set. If missing the default device seed will be set.
    enabled (bool): if ``False``, the RNG is not forked.  This is a convenience argument for easily disabling the context manager without having to delete it and unindent your Python code under it.
  """
  if not enabled:
    yield
    return

  if device is None:
    device = torch_xla._XLAC._xla_get_default_device()
  xla_rng_state = get_rng_state(device=device)

  try:
    yield
  finally:
    set_rng_state(xla_rng_state, device=device)


class MemoryInfo(TypedDict):
  bytes_used: str
  bytes_limit: int


def get_memory_info(device: Optional[torch.device] = None) -> MemoryInfo:
  """Retrieves the device memory usage.

  Args:
    device: Optional[torch.device] The device whose memory information are requested.
    If not passed will use the default device.

  Returns:
    MemoryInfo dict with memory usage for the given device.

  Example:

    >>> xm.get_memory_info()
    {'bytes_used': 290816, 'bytes_limit': 34088157184, 'peak_bytes_used': 500816}
  """
  if device == None:
    device = xla_device()
  return torch_xla._XLAC._xla_memory_info(str(device))


def optimization_barrier_(tensors: List[torch.Tensor]):
  """Blocks xla compiler from moving computations across this barrier. The common
  use case would be blocking xla common-subexpression elimination pass from undoing
  the gradient checkpointing.

  Args:
    tensors (List[torch.Tensor]): List of `torch.Tensor` to add barrier to.
  """
  torch_xla._XLAC._xla_optimization_barrier_(tensors)


def broadcast_master_param(model: torch.nn.Module) -> None:
  """
  Broadcast the model parameters from master process to other processes
  """
  parameters_and_buffers = list(
      itertools.chain(model.parameters(), model.buffers()))
  collective_broadcast(parameters_and_buffers)
  mark_step()
