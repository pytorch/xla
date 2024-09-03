import sys
import collections
import contextlib
import functools
import uuid
from typing import Any, Callable, List, Optional, Tuple
import weakref

import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_env_vars as xenv
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr
import torch_xla.utils.utils as xu


def device(index: int = None) -> torch.device:
  """Returns a given instance of an XLA device.

  If SPMD enables, returns a virtual device that wraps all devices available
  to this process.

  Args:
    index: index of the XLA device to be returned. Corresponds to index in
      `torch_xla.devices()`.

  Returns:
    An XLA `torch.device`.
  """

  return xm.xla_device(index)


def devices() -> List[torch.device]:
  """Returns all devices available in the current process.

  Returns:
    A list of XLA `torch.devices`.
  """

  return [torch.device(d) for d in xm.get_xla_supported_devices()]


def real_devices() -> List[str]:
  """Returns local XLA device types and indices.

  Returns:
    A list strings representing the XLA devices available in the current
    process, e.g. `['TPU:0', 'TPU:1', ...]`.
  """

  return torch_xla._XLAC._xla_real_devices()


def device_count() -> int:
  """Returns number of addressable devices in the current process."""
  return len(real_devices())


def sync(wait: bool = False):
  """Launches all pending graph operations.

  Args:
    wait (bool): whether to block the current process until the execution finished.

  """
  torch_xla._XLAC._xla_step_marker(
      torch_xla._XLAC._xla_get_default_device(),
      [],
      wait=wait,
  )
  devctx = xm._run_step_closures()
  torch_xla._XLAC._set_all_reduce_token(devctx.device, None)


def step():
  """Wraps code that should be dispatched to the runtime.

  Experimental: `xla.step` is still a work in progress. Some code that currently
  works with `xla.step` but does not follow best practices will become errors in
  future releases. See https://github.com/pytorch/xla/issues/6751 for context.
  """
  return compile()


# Keeps track of the alive functions. This allow us to remove session entries in the
# C++ side for functions that are no longer alive.
_compiled_id_to_functions_ref = weakref.WeakValueDictionary()


def compile(
    f: Optional[Callable] = None,
    full_graph: Optional[bool] = False,
    name: Optional[str] = None,
    num_different_graphs_allowed: Optional[int] = None,
):
  """
  Optimizes given model/function using torch_xla's LazyTensor tracing mode.
  PyTorch/XLA will trace the given function with given inputs and then generate
  graphs to represent the pytorch operations happens within this function. This
  graph will be compiled by the XLA and executed on the accelerator(decided by the
  tensor's device). Eager mode will be disabled for the compiled region of the funciton.

  Args:
      model (Callable): Module/function to optimize, if not passed this function will
        act as a context manager.
      full_graph (Optional[bool]): Whether this compile should generate a single graph. If set to True
        and multiple graphs will be generated torch_xla will throw an error with debug info
        and exit.
      name (Optional[name]): Name of the compiled program. The name of the function `f` will be used
        if not specified. This name will be used in the `PT_XLA_DEBUG` messages as well as HLO/IR dump
        file.
      num_different_graphs_allowed (Optional[int]): number of different traced graphs of the given
        model/function that we are allowed to have. An error will be raised in case this limit
        is exceeded.

  Example::

      # usage 1
      @torch_xla.compile()
      def foo(x):
        return torch.sin(x) + torch.cos(x)

      def foo2(x):
        return torch.sin(x) + torch.cos(x)
      # usage 2
      compiled_foo2 = torch_xla.compile(foo2)

      # usage 3
      with torch_xla.compile():
        res = foo2(x)
  """
  if name is None and f is not None:
    if hasattr(f, '__name__'):
      name = f.__name__
    elif hasattr(f, '__str__'):
      name = f.__str__()

  if f is not None:
    current_id = f"{name}_{id(f)}"
  else:
    current_id = str(uuid.uuid4())

  # Check whether the function/module that corresponds with current_id is still alive. If it's not,
  # we can remove it from the session's map in the C++ side, so we can start a fresh session.
  #
  # This solves the issue where there are 2 different local-scoped functions with the same name.
  # Since they are local-scoped, they might end-up with the same id. And, since they have the same
  # name, their current_id will be the same, even though they are different functions.
  #
  # This issue was observed when running test_dynamic_shape_detector.py.
  if current_id not in _compiled_id_to_functions_ref:
    torch_xla._XLAC._dynamic_shape_detector_remove_session(current_id)

  if f is not None:
    _compiled_id_to_functions_ref[current_id] = f

  def _clear_pending_ops_before_compile():
    sync()

  @contextlib.contextmanager
  def _compile():
    saved_eager_mode_status = torch_xla._XLAC._get_use_eager_mode()
    saved_allow_execution = torch_xla._XLAC._get_allow_execution()
    saved_current_graph_name = torch_xla._XLAC._get_current_graph_name()
    torch_xla._XLAC._set_use_eager_mode(False)
    if name is not None:
      torch_xla._XLAC._set_current_graph_name(name + '_clear_pending')
    # Clear pending operations
    _clear_pending_ops_before_compile()

    if name is not None:
      torch_xla._XLAC._set_current_graph_name(name)

    # if full_graph sets to true execution can not happen before the sync below
    torch_xla._XLAC._set_allow_execution(not full_graph)

    if num_different_graphs_allowed is not None:
      torch_xla._XLAC._dynamic_shape_detector_set_max_num_different_graphs_allowed(
          num_different_graphs_allowed)
      torch_xla._XLAC._dynamic_shape_detector_start_session(current_id)

    try:
      yield
    finally:
      torch_xla._XLAC._set_allow_execution(saved_allow_execution)
      if num_different_graphs_allowed is not None:
        torch_xla._XLAC._dynamic_shape_detector_end_session()
      # Collect the traced graph after running the target function and
      # execute the graph.
      sync()
      torch_xla._XLAC._set_use_eager_mode(saved_eager_mode_status)
      torch_xla._XLAC._set_current_graph_name(saved_current_graph_name)

  return _compile() if f is None else _compile()(f)


def manual_seed(seed, device=None):
  """Set the seed for generating random numbers for the current XLA device.

  Args:
    seed (integer): The state to be set.
    device (torch.device, optional): The device where the RNG state needs to be set.
      If missing the default device seed will be set.
  """
  xm.set_rng_state(seed, device)


# TODO(wcromar): Update args to type ParamSpec.
def launch(
    fn: Callable,
    args: Tuple = (),
    start_method: str = 'spawn',
    debug_single_process: bool = False,
):
  """ Entry to launch multiprocess.

  Raises:
    NotImplementedError: SPMD is not supported yet.
  """
  if xr.is_spmd():
    # TODO(piz): SPMD is specified differently from mp. Skip for now.
    raise NotImplementedError(
        'launch function does not support SPMD at this time')

  nprocs = 1 if debug_single_process else None

  if dist.is_torchelastic_launched():
    fn(xu.getenv_as(xenv.LOCAL_RANK, int), *args)
  else:
    xmp.spawn(fn, args=args, nprocs=nprocs, start_method=start_method)
