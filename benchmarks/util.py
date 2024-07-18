import argparse
from contextlib import contextmanager
import functools
import logging
import os
from os.path import abspath, exists
import subprocess
import torch
import torch.utils._pytree as pytree
from typing import Any, Union
import sys
import torch_xla.core.xla_model as xm
from torch_xla._internal import tpu

logger = logging.getLogger(__name__)

StrOrBool = Union[str, bool]


def parse_none_str(a: Any):
  if isinstance(a, str) and a.upper() == "None".upper():
    return None
  return a


def ns_to_s(ns):
  return ns * 1e-9


def us_to_s(us):
  return us * 1e-6


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


@functools.lru_cache(maxsize=3)
def is_xla_device_available(devkind, use_xla2: bool = False):
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
  r = subprocess.run([sys.executable, python_file, devkind, str(use_xla2)])
  return r.returncode == 0


def move_to_device(item, device, torch_xla2: bool = False):
  if torch_xla2:
    import torch_xla2
    import jax
    move_to_device_func = lambda t: jax.device_put(
        torch_xla2.tensor.t2j(t), device)
  else:
    move_to_device_func = lambda t: t.to(device)
  return pytree.tree_map_only(torch.Tensor, move_to_device_func, item)


def cast_to_dtype(item: Any, dtype: torch.dtype) -> Any:
  return pytree.tree_map_only(
      torch.Tensor,
      lambda t: t.to(dtype)
      if isinstance(t, torch.Tensor) and t.is_floating_point() else t,
      item,
  )


def randomize_input(inputs: Any):
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
def set_cwd(path: str):
  original_dir = abspath(os.getcwd())
  os.chdir(path)
  try:
    yield
  finally:
    os.chdir(original_dir)


def get_accelerator_model(accelerator: str):
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


def get_torchbench_test_name(test):
  return {"train": "training", "eval": "inference"}[test]


def find_near_file(names: str):
  """Find a file near the current directory.

  Looks for `names` in the current directory, up to its two direct parents.
  """
  for dir in ("./", "../", "../../", "../../../"):
    for name in names:
      path = os.path.join(dir, name)
      if exists(path):
        return abspath(path)
  return None


def reset_rng_state(benchmark_experiment: "BenchmarkExperiment"):
  import numpy as np
  import random
  SEED = 1337
  torch.manual_seed(SEED)
  random.seed(SEED)
  np.random.seed(SEED)
  # TODO(piz): setup the rng state on jax for torch_xla2.
  if benchmark_experiment.xla is not None and benchmark_experiment.torch_xla2 is None:
    device = benchmark_experiment.get_device()
    xm.set_rng_state(SEED, str(device))
