import torch.multiprocessing
import torch_xla
from torch_xla import runtime as xr
from torch_xla._internal import pjrt

from functools import cache
from typing import Union, Sequence, Any, Optional, List, Callable


def spawn(fn,
          args=(),
          nprocs=None,
          join=True,
          daemon=False,
          start_method='spawn'):
  """Enables multi processing based replication.

  Args:
    fn (callable): The function to be called for each device which takes part of
      the replication. The function will be called with a first argument being
      the global index of the process within the replication, followed by the
      arguments passed in `args`.
    args (tuple): The arguments for `fn`.
      Default: Empty tuple
    nprocs (int): The number of processes/devices for the replication. At the
      moment, if specified, can be either 1 or None (which would automatically
      converted to the maximum number of devices). Other numbers would result
      in ValueError.
    join (bool): Whether the call should block waiting for the completion of the
      processes which have being spawned.
      Default: True
    daemon (bool): Whether the processes being spawned should have the `daemon`
      flag set (see Python multi-processing API).
      Default: False
    start_method (string): The Python `multiprocessing` process creation method.
      Default: `spawn`

  Returns:
    The same object returned by the `torch.multiprocessing.spawn` API. If
    `nprocs` is 1 the `fn` function will be called directly, and the API will
    return None.
  """
  return pjrt.spawn(fn, nprocs, start_method, args)


class MpModelWrapper(object):
  """Wraps a model to minimize host memory usage when `fork` method is used.

  This class should be used together with the `spawn(..., start_method='fork')`
  API to minimize the use of host memory.
  Instead of creating models on each multiprocessing process, hence replicating
  the model's initial host memory, the model is created once at global scope,
  and then moved into each device inside the `spawn()` target function.
  Example::

    WRAPPED_MODEL = xmp.MpModelWrapper(MyNetwork())

    def _mp_fn(index, ...):
      device = xm.xla_device()
      model = WRAPPED_MODEL.to(device)
      ...

    torch_xla.launch(_mp_fn, ..., start_method='fork')

  This method has two advantages. First it uses only one copy of the memory
  pages to host the original model weights, and second it serializes the move
  of the wrapped model into each device, by lowering the load onto the system
  memory during the process.
  """

  def __init__(self, model):
    """Creates a new `MpModelWrapper` object.

    Args:
      model (torch.nn.Module): The model to be wrapped. Should be on PyTorch CPU
        device (which is the default when creating new models).
    """
    self._model = model
    self._lock = torch.multiprocessing.Lock()

  def to(self, device):
    """Retrieves the model moved onto the specified device.

    Args:
      device (torch.device): The device where the model should be moved onto.
    Returns:
      The model on the specified device.
    """
    with self._lock:
      self._model.to(device)
    return self._model


class MpSerialExecutor(object):
  """Utility to run a function in a serialized fashion among multi-core processes.

  Example::

    # At global scope.
    SERIAL_EXEC = xmp.MpSerialExecutor()

    def load_dataset(path):
      return maybe_download_and_load(path)

    def _mp_fn(index, ...):
      # Avoid all cores downloading the same data with the serial executor.
      dataset = SERIAL_EXEC.run(lambda: load_dataset('/tmp/mnist-data'))
      ...

    torch_xla.launch(_mp_fn, ...)
  """

  def __init__(self):
    self._lock = torch.multiprocessing.Lock()

  def run(self, fn):
    """Runs the provided function serialized WRT each per-core process.

    Args:
      fn (callable): The function to run in a serialized fashion.
    Returns:
      The `fn` return value.
    """
    with self._lock:
      return fn()


###############################################################################
#
# The following is modified from JAX: https://github.com/jax-ml/jax/blob/main/jax/_src/mesh_utils.py
#
###############################################################################

_TPU_V5P = "v5p"
_TPU_V6E = "v6e"

_V5P_2x2x2_ORDER = (0, 1, 3, 2, 6, 7, 5, 4)
_V6E_2x4_ORDER = (0, 2, 4, 6, 7, 5, 3, 1)


@cache
def _get_xyz_bounds():
  devices = xr.global_runtime_device_attributes()
  max_x, max_y, max_z = max(tuple(d.get("coords", (0, 0, 0))) for d in devices)
  bound_x, bound_y, bound_z = max_x + 1, max_y + 1, max_z + 1
  return bound_x, bound_y, bound_z


def _v5p_create_replica_groups() -> List | None:
  """Creates optimized replica groups order for selected topologies.

  Returns:
    None or reordered replica groups.
  """
  bound_x, bound_y, bound_z = _get_xyz_bounds()

  if bound_x == bound_y == 2 and bound_z == 2:
    replica_group = list(_V5P_2x2x2_ORDER)
    return [replica_group]
  return None


def _v6e_create_replica_groups() -> List | None:
  """Creates optimized replica groups order for selected topologies.

  Returns:
    None or reordered replica groups.
  """
  bound_x, bound_y, bound_z = _get_xyz_bounds()

  if bound_x == 2 and bound_y == 4:
    replica_group = list(_V6E_2x4_ORDER)
    return [replica_group]
  return None


device_kind_handler_dict: dict[
    str,
    Callable[..., List | None],
] = {
    _TPU_V5P: _v5p_create_replica_groups,
    _TPU_V6E: _v6e_create_replica_groups
}


def create_optimized_replica_groups() -> List | None:
  """Creates optimized replica groups order for different TPU types.

  Returns:
    None or reordered replica groups.
  """
  tpu_type = torch_xla.tpu.get_tpu_type()
  handler = device_kind_handler_dict.get(tpu_type, None)
  if handler is not None:
    return handler()
  return None
