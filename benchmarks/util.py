from contextlib import contextmanager
import functools
import logging
from multiprocessing import Process, Queue
import numpy as np
import os
from os.path import abspath
import queue
import random
import torch
import traceback

logger = logging.getLogger(__name__)


@functools.lru_cache(None)
def patch_torch_manual_seed():
  """Make torch manual seed deterministic. Helps with accuracy testing."""

  def deterministic_torch_manual_seed(*args, **kwargs):
    from torch._C import default_generator

    seed = 1337
    import torch.cuda

    if not torch.cuda._is_in_bad_fork():
      torch.cuda.manual_seed_all(seed)
    return default_generator.manual_seed(seed)

  torch.manual_seed = deterministic_torch_manual_seed


def reset_rng_state(benchmark_experiment=None):
  torch.manual_seed(1337)
  random.seed(1337)
  np.random.seed(1337)
  if benchmark_experiment and benchmark_experiment.xla:
    import torch_xla.core.xla_model as xm
    device = benchmark_experiment.get_device()
    xm.set_rng_state(1337, str(device))


@functools.lru_cache(maxsize=3)
def is_xla_device_available(devkind):
  if devkind not in ["CPU", "GPU", "TPU"]:
    raise ValueError(devkind)

  def _check_xla_device(q, devkind):
    try:
      import os
      os.environ["PJRT_DEVICE"] = devkind

      import torch_xla.core.xla_model as xm

      q.put(bool(xm.get_xla_supported_devices(devkind=devkind)))
    except Exception:
      traceback.print_exc()
      q.put(False)

  q = Queue()
  process = Process(target=_check_xla_device, args=(q, devkind))
  process.start()
  process.join(60)
  try:
    return q.get_nowait()
  except queue.Empty:
    traceback.print_exc()
    return False


def move_to_device(item, device):
  if isinstance(item, torch.Tensor):
    return item.to(device=device)
  elif isinstance(item, list):
    return [move_to_device(t, device) for t in item]
  elif isinstance(item, tuple):
    return tuple(move_to_device(t, device) for t in item)
  elif isinstance(item, dict):
    return dict((k, move_to_device(t, device)) for k, t in item.items())
  else:
    return item


def randomize_input(inputs):
  if isinstance(inputs, torch.Tensor):
    if inputs.dtype in (torch.float32, torch.float64):
      torch._dynamo.utils.counters["randomize_input"]["times"] += 1
      return torch.randn_like(inputs)
    elif inputs.dtype == torch.int64:
      # Note: we can not simply tune integer tensors as follows
      #   `return torch.randint_like(inputs, high=inputs.max().item())`
      # This may break some invariants between tensors.
      # E.g. in embedding lookup case, one tensor is the length
      # and another is an indices tensor.
      return inputs
    else:
      raise RuntimeError(
          f"randomize_input need support tensor of type {inputs.dtype}"
      )
  elif isinstance(inputs, (list, tuple)):
    return type(inputs)([randomize_input(x) for x in inputs])
  elif isinstance(inputs, dict):
    return dict((k, randomize_input(x)) for k, x in inputs.items())
  else:
    logger.warning(
        f"randomize_input can not handle input of type {type(inputs)}"
    )
    return inputs


@contextmanager
def set_cwd(path):
  original_dir = abspath(os.getcwd())
  os.chdir(path)
  try:
    yield
  finally:
    os.chdir(original_dir)