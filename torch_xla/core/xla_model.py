import io
import itertools
import logging
import sys
import re
import threading
import time
from typing import List, Optional
import torch
import torch.distributed._functional_collectives
import torch.nn.functional as F
import torch_xla
from torch_xla import runtime
import torch_xla.core.xla_env_vars as xenv
import torch_xla.debug.metrics_saver as ms
import torch_xla.utils.utils as xu
import torch_xla.utils.closures as xc

_DEVICES = xu.LazyProperty(lambda: torch_xla._XLAC._xla_get_devices())

REDUCE_SUM = 'sum'
REDUCE_MUL = 'mul'
REDUCE_AND = 'and'
REDUCE_OR = 'or'
REDUCE_MIN = 'min'
REDUCE_MAX = 'max'

_DEVICE_CONTEXTS = dict()
_DEVICE_CONTEXTS_LOCK = threading.Lock()

# Note [Dynamo WORLD_SIEZ and ORDINAL]
# Belows are workaround to cache the ordinal and world_size such that
# Dynamo won't do graph breaks when xm.xrt_world_size() and xm.get_ordinal() are called.
_WORLD_SIZE = None
_ORDINAL = None


def _init_world_size_ordinal():
  global _WORLD_SIZE, _ORDINAL

  # Dynamo doesn't support XRT or multithreaded runtime. See Note [V3-8 Threading]
  if not runtime.using_pjrt() or runtime.addressable_device_count() > 1:
    return

  if _WORLD_SIZE is None:
    _WORLD_SIZE = xrt_world_size()
    _ORDINAL = get_ordinal()


class DeviceContext(object):

  def __init__(self, device):
    self.device = device


def _get_device_context(device=None):
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


def is_xla_tensor(tensor):
  return tensor.device.type == 'xla'


def parse_xla_device(device):
  m = re.match(r'(CPU|TPU|GPU|XPU|NEURON):(\d+)$', device)
  if m:
    return (m.group(1), int(m.group(2)))


def get_xla_supported_devices(devkind=None, max_devices=None):
  """Returns a list of supported devices of a given kind.

  Args:
    devkind (string..., optional): If specified, one of `TPU`, `GPU`, `XPU`, 
      `NEURON` or `CPU` (the 'GPU' XLA device is currently not implemented).
    max_devices (int, optional): The maximum number of devices to be returned of
      that kind.

  Returns:
    The list of device strings.
  """
  xla_devices = _DEVICES.value
  devkind = [devkind] if devkind else ['TPU', 'GPU', 'XPU', 'NEURON', 'CPU']
  for kind in devkind:
    kind_devices = []
    for i, device in enumerate(xla_devices):
      if re.match(kind + r':\d+$', device):
        kind_devices.append('xla:{}'.format(i))
    if kind_devices:
      return kind_devices[:max_devices] if max_devices else kind_devices


def xrt_world_size(defval=1):
  """Retrieves the number of devices which is taking part of the replication.

  Args:
    defval (int, optional): The default value to be returned in case there is no
      replication information available.
      Default: 1

  Returns:
    The number of devices which is taking part of the replication.
  """
  global _WORLD_SIZE
  if _WORLD_SIZE is not None:
    return _WORLD_SIZE

  return runtime.world_size()


def get_ordinal(defval=0):
  """Retrieves the replication ordinal of the current thread.

  The ordinals range from 0 to `xrt_world_size()` minus 1.

  Args:
    defval (int, optional): The default value to be returned in case there is no
      replication information available. Ignored for runtime.
      Default: 0

  Returns:
    The replication ordinal of the current thread.
  """
  global _ORDINAL
  if _ORDINAL is not None:
    return _ORDINAL

  return runtime.global_ordinal()


def get_local_ordinal(defval=0):
  """Retrieves the replication local ordinal of the current thread.

  The local ordinals range from 0 to the number of local devices minus 1.

  Args:
    defval (int, optional): The default value to be returned in case there is no
      replication information available. Ignored for runtime.
      Default: 0

  Returns:
    The replication local ordinal of the current thread.
  """
  return runtime.local_ordinal()


def is_master_ordinal(local=True):
  """Checks whether the current process is the master ordinal (0).

  Args:
    local (bool): Whether the local or global master ordinal should be checked.
      In case of multi-host replication, there is only one global master ordinal
      (host 0, device 0), while there are NUM_HOSTS local master ordinals.
      Default: True

  Returns:
    A boolean indicating whether the current process is the master ordinal.
  """
  ordinal = get_local_ordinal() if local else get_ordinal()
  return ordinal == 0


def master_print(*args, fd=sys.stdout, local=False, flush=False):
  if is_master_ordinal(local=local):
    print(*args, file=fd, flush=flush)


def xla_device(n=None, devkind=None):
  """Returns a given instance of an XLA device.

  Args:
    n (int, optional): The specific instance (ordinal) to be returned. If
      specified, the specific XLA device instance will be returned. Otherwise
      the first device of `devkind` will be returned.
    devkind (string..., optional): If specified, one of `TPU`, `GPU`, `XPU` 
      `NEURON` or `CPU`.

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


def _xla_real_device(device):
  device_str = str(device)
  m = re.match(r'xla:(\d+)$', device_str)
  if not m:
    raise RuntimeError('Invalid device format: {}'.format(device_str))
  return _DEVICES.value[int(m.group(1))]


def xla_real_devices(devices):
  return [_xla_real_device(device) for device in devices]


def xla_device_hw(device):
  """Returns the hardware type of the given device.

  Args:
    device (string or torch.device): The xla device that will be mapped to the
      real device.

  Returns:
    A string representation of the hardware type (`CPU`, `TPU`, `XPU`, `NEURON`, `GPU`) 
    of the given device.
  """
  real_device = _xla_real_device(device)
  return real_device.split(':')[0]


def xla_replication_devices(local_devices):
  real_devices = xla_real_devices(local_devices)
  device_types = set()
  for device in real_devices:
    xdev = parse_xla_device(device)
    device_types.add(xdev[0])
  if len(device_types) != 1:
    # No replication if the device set spawns multiple device types.
    raise RuntimeError(
        'Cannot replicate across different device types: devices={}/{}'.format(
            local_devices, real_devices))
  device_type = device_types.pop()
  kind_devices = get_xla_supported_devices(devkind=device_type)
  if len(kind_devices) != len(local_devices):
    # Replication can only happen among all devices of one kind.
    raise RuntimeError(
        'Cannot replicate if number of devices ({}) is different from {}'.
        format(len(local_devices), len(kind_devices)))
  replication_devices = []
  for device in torch_xla._XLAC._xla_get_all_devices():
    xdev = parse_xla_device(device)
    if not xdev:
      raise RuntimeError('Invalid device format: {}'.format(device))
    if xdev[0] == device_type:
      replication_devices.append(device)
  sorted_by_ordinal = sorted(
      replication_devices, key=lambda device: parse_xla_device(device)[1])
  return sorted_by_ordinal


def unlazy(tensors):
  """Blocks the program until `tensors` are materialized.

  This API is for benchmarking, don't use it in real models.

  Args:
    tensors (List[torch.Tensor]): List of `torch.Tensor`s to materialize. For
      each Tensor `t` in the list, `t.device` must be an `xla` device.
  """
  torch_xla._XLAC._xla_sync_multi(tensors, devices=[], wait=True)


def set_replication(device, devices):
  device = str(device)
  devctx = _get_device_context(device=device)
  devices = [str(x) for x in devices]
  if devices:
    replication_devices = xla_replication_devices(devices)
    torch_xla._XLAC._xla_set_replication_devices(replication_devices)
    devctx.device_index = devices.index(device)
  else:
    torch_xla._XLAC._xla_set_replication_devices([])
    devctx.device_index = 0
  torch_xla._XLAC._set_all_reduce_token(devctx.device, None)
  torch_xla._XLAC._xla_set_default_device(device)


class RateTracker(object):

  def __init__(self, smooth_factor=None):
    self._smooth_factor = xu.getenv_as(
        'RATE_TRACKER_SMOOTHING', float,
        0.4) if smooth_factor is None else smooth_factor
    self._start_time = time.time()
    self._partial_time = self._start_time
    self._partial_count = 0.0
    self._partial_rate = None
    self._count = 0.0

  def _update(self, now, rate):
    self._partial_count += self._count
    self._count = 0.0
    self._partial_time = now
    self._partial_rate = rate

  def add(self, count):
    self._count += count

  def _smooth(self, current_rate):
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

  def __init__(self, convert_fn, select_fn):
    self._convert_fn = convert_fn
    self._select_fn = select_fn
    self._tensors = []

  def _add(self, tensor):
    self._tensors.append(tensor)

  def _convert(self):
    self._index = 0
    if self._tensors:
      self._converted_tensors = self._convert_fn(self._tensors)
    else:
      self._converted_tensors = []

  def _get_converted_tensor(self):
    assert self._index < len(self._converted_tensors)
    new_tensor = self._converted_tensors[self._index]
    self._index += 1
    return new_tensor

  def _collect_tensors(self, inputs):

    def collect_fn(value):
      self._add(value)

    xu.for_each_instance(inputs, lambda x: self._select_fn(x), collect_fn)

  def _replace_tensors(self, inputs):

    def convert_fn(value):
      return self._get_converted_tensor()

    return xu.for_each_instance_rewrite(inputs, lambda x: self._select_fn(x),
                                        convert_fn)

  def transform(self, inputs):
    self._tensors = []
    self._collect_tensors(inputs)
    self._convert()
    return self._replace_tensors(inputs)


def check_view_sharing(obj):
  tensors = set()
  aliases = dict()

  def tensor_info(t):
    return '{}{}'.format(t.dtype, list(t.size()))

  def tensor_id(t):
    if is_xla_tensor(t):
      return torch_xla._XLAC._xla_get_tensor_id(t), 'xla'
    return id(t), 'torch'

  def alias_id(t):
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


def _fetch_gradients(optimizer):
  gradients = []
  for param_group in optimizer.__getstate__()['param_groups']:
    for group, params in param_group.items():
      if group == 'params':
        for p in params:
          if isinstance(p, torch.Tensor) and p.grad is not None:
            gradients.append(p.grad.data)
  return gradients


def _get_all_reduce_token():
  devctx = _get_device_context()
  token = torch_xla._XLAC._get_all_reduce_token(devctx.device)
  return token, devctx


def all_reduce(reduce_type, inputs, scale=1.0, groups=None, pin_layout=True):
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
  if xrt_world_size() == 1 and not xu.getenv_as('XLA_ALWAYS_ALLREDUCE', bool,
                                                False):
    if isinstance(inputs, torch.Tensor):
      return inputs.clone()
    else:
      return inputs

  if isinstance(inputs, torch.Tensor):
    result = None
    if scale == 1.0 and groups == [] and pin_layout:
      # TODO(alanwaketan): Support groups.
      # Only c10d_functional version cc ops are traceable by Dynamo.
      result = torch.ops.c10d_functional.all_reduce(inputs, reduce_type, "", [],
                                                    0)
    else:
      result = torch_xla._XLAC._xla_all_reduce(reduce_type, inputs, scale,
                                               groups, pin_layout)
    results = [result]
  else:
    torch_xla._XLAC._xla_all_reduce_inplace(reduce_type, inputs, scale, groups,
                                            pin_layout)
    results = inputs

  return results[0] if isinstance(inputs, torch.Tensor) else results


def _all_gather_using_all_reduce(value, dim=0, groups=None, pin_layout=True):
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
  if dim < 0:
    dim = value.dim() + dim
  size = value.size(dim)
  padding = [0] * (2 * value.dim())
  ordinal = get_ordinal()
  if groups is None:
    left, right = ordinal, xrt_world_size() - 1 - ordinal
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


def all_gather(value, dim=0, groups=None, output=None, pin_layout=True):
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
  if pin_layout and (output == None or xla_device_hw(value.device) == 'NEURON'):
    # There is not an easy way to pin the all_gather layout on TPU, GPU and NEURON,
    # use all_reduce based all_gather for this purpose.
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
    shard_count = xrt_world_size()

  token, devctx = _get_all_reduce_token()
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


def all_to_all(value,
               split_dimension,
               concat_dimension,
               split_count,
               groups=None,
               pin_layout=True):
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


def collective_permute(value, pairs):
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
          1 if get_ordinal() == root_ordinal else 0, dtype=tensor.dtype)
      # Transfer scale tensor as device data instead of constant 1 or 0.
      xscale = send_cpu_data_to_device(scale, tensor.device)
      tensor.mul_(xscale[0])

  all_reduce(REDUCE_SUM, tensors, groups=groups, pin_layout=pin_layout)


def send(value, channel_id):
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


def recv(output, channel_id):
  """Performs a XLA `Send()` operation on the input tensor.

  See: https://www.tensorflow.org/xla/operation_semantics#recv

  Args:
    output (torch.Tensor): The output tensor.
    channel_id (int64): opaque id identifying the source of the recv op.
  """
  token, devctx = _get_all_reduce_token()
  result, new_token = torch_xla._XLAC._xla_recv(output, token, channel_id)
  torch_xla._XLAC._set_all_reduce_token(devctx.device, new_token)
  return result


def reduce_scatter(reduce_type,
                   input,
                   scale,
                   scatter_dim,
                   shard_count,
                   groups=None,
                   output=None,
                   pin_layout=True):
  """Performs a XLA `ReduceScatter()` operation on the input tensor.

  See: https://www.tensorflow.org/xla/operation_semantics#reducescatter

  Args:
    reduce_type (string): One of ``xm.REDUCE_SUM``, ``xm.REDUCE_MUL``,
      ``xm.REDUCE_AND``, ``xm.REDUCE_OR``, ``xm.REDUCE_MIN`` and
      ``xm.REDUCE_MAX``.
    input: A single `torch.Tensor` all reduce + scatter op to.
    scale (float): A default scaling value to be applied after the reduce.
    scatter_dim (int): Dimension number to which apply scatter operation.
    shard_count (int): The number of ways to split up the scatter_dim in.
    groups (list): A list of list, representing the replica groups for
      the `all_reduce()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
        defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
        the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
        all the replicas in it.
    output: Optional output tensor
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
  if output != None:
    # Call the out of place version of the reduce_scatter
    new_token = torch_xla._XLAC._xla_reduce_scatter_out(reduce_type, output,
                                                        input, token, scale,
                                                        scatter_dim,
                                                        shard_count, groups or
                                                        [], pin_layout)
    torch_xla._XLAC._set_all_reduce_token(devctx.device, new_token)
    return output

  result = torch_xla._XLAC._xla_reduce_scatter(reduce_type, input, token, scale,
                                               scatter_dim, shard_count,
                                               groups or [], pin_layout)
  torch_xla._XLAC._set_all_reduce_token(devctx.device, result[1])
  return result[0]


def add_step_closure(closure, args=(), run_async=False):
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


def _run_step_closures():
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


def mark_step(wait=False):
  if xu.getenv_as('XLA_EMIT_STEPLOG', bool, False):
    print(
        'torch_xla.core.xla_model::mark_step\n',
        end='',
        file=sys.stderr,
        flush=True)
  torch_xla._XLAC._xla_step_marker(
      torch_xla._XLAC._xla_get_default_device(), [],
      wait=xu.getenv_as('XLA_SYNC_WAIT', bool, wait))
  # Only emit metrics from the first local device index, to avoid emitting the
  # same values from different threads.
  if is_master_ordinal():
    ms.save_metrics()
  devctx = _run_step_closures()
  torch_xla._XLAC._set_all_reduce_token(devctx.device, None)


def get_stablehlo(tensors=None) -> str:
  """Get StableHLO for the computation graph in string format.

  If `tensors` is not empty, the graph with `tensors` as outputs will be dump.
  If `tensors` is empty, the whole computation graph will be dump.
  TODO(lsy323): When `tensors` is empty, the some intermediate tensors will also be
  dump as outputs. Need further investigation.

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


def get_stablehlo_bytecode(tensors=None) -> bytes:
  """Get StableHLO for the computation graph in bytecode format.

  If `tensors` is not empty, the graph with `tensors` as outputs will be dump.
  If `tensors` is empty, the whole computation graph will be dump.
  TODO(lsy323): When `tensors` is empty, the some intermediate tensors will also be
  dump as outputs. Need further investigation.

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


def wait_device_ops(devices=[]):
  """Waits for all the async operations on the given devices to complete.

  Args:
    devices (string..., optional): The devices whose async ops need to be waited
      for. If empty, all the local devices will be waited for.
  """
  torch_xla._XLAC._xla_wait_device_ops(devices=devices)


def reduce_gradients(optimizer, groups=None, pin_layout=True):
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
  count = xrt_world_size()
  if count > 1:
    gradients = _fetch_gradients(optimizer)
    all_reduce(
        REDUCE_SUM,
        gradients,
        scale=1.0 / count,
        groups=groups,
        pin_layout=pin_layout)


def optimizer_step(optimizer,
                   barrier=False,
                   optimizer_args={},
                   groups=None,
                   pin_layout=True):
  """Run the provided optimizer step and issue the XLA device step computation.

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
  """
  reduce_gradients(optimizer, groups=groups, pin_layout=pin_layout)
  loss = optimizer.step(**optimizer_args)
  if barrier:
    mark_step()
  return loss


def save(data, file_or_path, master_only=True, global_master=False):
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
    sync (bool, optional): Whether to synchronize all replicas after saving
      tensors. If True, all replicas must call `xm.save` or the main process
      will hang.
  """
  should_write_data = not master_only or is_master_ordinal(
      local=not global_master)

  cpu_data = _maybe_convert_to_cpu(data, convert=should_write_data)
  if should_write_data:
    torch.save(cpu_data, file_or_path)


def _maybe_convert_to_cpu(data, convert=True):

  def convert_fn(tensors):
    torch_xla._XLAC._xla_sync_multi(
        tensors, devices=[], wait=True, sync_xla_data=True)
    if not convert:
      return tensors
    return torch_xla._XLAC._xla_get_cpu_tensors(tensors)

  def select_fn(v):
    return type(v) == torch.Tensor and is_xla_tensor(v)

  return ToXlaTensorArena(convert_fn, select_fn).transform(data)


def send_cpu_data_to_device(datas, device, input_sharding=None):

  def convert_fn(tensors):
    devices = [str(device)] * len(tensors)
    shardings = None
    if input_sharding:
      shardings = [input_sharding.xla_spec(t) for t in tensors]
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


def rendezvous(tag, payload=b'', replicas=[]):
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
  """
  return xla_rendezvous(payload, replicas or None, tag=tag)


def do_on_ordinals(target, data=(), ordinals=(0,)):
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
  running = get_ordinal() in ordinals
  cpu_data = _maybe_convert_to_cpu(data, convert=running)
  if running:
    result = target(*cpu_data)
  else:
    result = None
  rendezvous('torch_xla.core.xla_model.do_on_ordinals')
  return result


def mesh_reduce(tag, data, reduce_fn):
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


def set_rng_state(seed, device=None):
  """Sets the random number generator state.

  Args:
    seed (integer): The state to be set.
    device (string, optional): The device where the RNG state needs to be set.
      If missing the default device seed will be set.
  """
  if device is None:
    device = torch_xla._XLAC._xla_get_default_device()
  torch_xla._XLAC._xla_set_rng_seed(seed, str(device) if device else '')


def get_rng_state(device=None):
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


def get_memory_info(device):
  """Retrieves the device memory information.

  Args:
    device (string): The device whose memory information are requested.

  Returns:
    A dictionary with `kb_free` (free memory in KB) and `kb_total` (total
    memory in KB) keys.
  """
  return torch_xla._XLAC._xla_memory_info(str(device))


def optimization_barrier_(tensors):
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
