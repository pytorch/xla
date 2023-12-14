from contextlib import contextmanager
import functools
import logging
import numpy as np
import os
from os.path import abspath
import random
import subprocess
import torch
import sys
import torch_xla.core.xla_model as xm
from torch_xla._internal import tpu

logger = logging.getLogger(__name__)


def parse_none_str(a: str):
  if isinstance(a, str) and a.upper() == "None".upper():
    return None
  return a


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
    device = benchmark_experiment.get_device()
    xm.set_rng_state(1337, str(device))


@functools.lru_cache(maxsize=3)
def is_xla_device_available(devkind):
  if devkind not in ["CPU", "CUDA", "TPU"]:
    raise ValueError(devkind)
  # Checking the availability of a given device kind.
  #
  # We intentionally use subprocess instead of multiprocessing library. The
  # reason being that we might initialize CUDA in the parent process and use
  # CUDA in the child process. This is a known limitation of using CUDA and
  # forking the process.
  #
  # In this case, subprocess works because it replaces the forked memory with
  # the execution of the new program (fresh memory), avoiding the error.
  #
  # For more information: https://github.com/pytorch/xla/pull/5960
  CHECK_XLA_DEVICE_PY = "check_xla_device.py"
  python_file = os.path.join(os.path.dirname(__file__), CHECK_XLA_DEVICE_PY)
  r = subprocess.run([sys.executable, python_file, devkind])
  return r.returncode == 0


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
          f"randomize_input need support tensor of type {inputs.dtype}")
  elif isinstance(inputs, (list, tuple)):
    return type(inputs)([randomize_input(x) for x in inputs])
  elif isinstance(inputs, dict):
    return dict((k, randomize_input(x)) for k, x in inputs.items())
  else:
    logger.warning(
        f"randomize_input can not handle input of type {type(inputs)}")
    return inputs


@contextmanager
def set_cwd(path):
  original_dir = abspath(os.getcwd())
  os.chdir(path)
  try:
    yield
  finally:
    os.chdir(original_dir)


def get_accelerator_model(accelerator):
  if accelerator == "cpu":
    return get_cpu_name()
  elif accelerator == "cuda":
    return get_gpu_name()
  elif accelerator == "tpu":
    return get_tpu_name()
  else:
    raise NotImplementedError


def get_cpu_name():
  return subprocess.check_output(
      ["lscpu"],
      encoding='utf-8').split("Model name:")[1].split("\n")[0].strip()


def get_gpu_name():
  gpu_names = subprocess.check_output(
      ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"],
      encoding='utf-8').split("\n")[1:-1]
  if len(gpu_names) == 1:
    return gpu_names[0]
  return "One of " + ", ".join(gpu_names)


def get_tpu_name():
  return tpu._get_metadata("accelerator-type")
