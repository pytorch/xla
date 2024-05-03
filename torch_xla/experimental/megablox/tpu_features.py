"""Module with utilities to detect TPU features."""
# TODO(shabalin): We will need to generalize this to support GPUs.
import re
from absl import flags

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla._internal import tpu


def is_tpu() -> bool:
  return "TPU" in torch_xla.core.xla_model.xla_device_hw(xm.xla_device())


def tpu_kind() -> str:
  """Query identification string for the currently attached TPU.

  Returns:
    "TPU v2"
    "TPU v3"
    "TPU v4"
    "TPU v5"
    "TPU v5e"
    "TPU v5p"
  """
  return torch_xla.core.xla_model.xla_device_hw(xm.xla_device()) + ' v' + str(
      tpu.version())


def tpu_config_name() -> str:
  """Query configuration name of the currently attached TPU.

  Returns:
    "" if no configuration name was specified.
    "megacore" if megacore configuration is specified.
  """
  return flags.FLAGS.deepsea_chip_config_name


_TPU_KIND_PATTERN = re.compile(r"TPU v(\d+)")


def tpu_generation() -> int:
  """Generation number of the currently attached TPU."""
  if version := _TPU_KIND_PATTERN.match(tpu_kind()):
    return int(version[1])
  raise NotImplementedError("only TPU devices are supported")


def supports_bfloat16_matmul() -> bool:
  """Does the currently attached CPU supports bfloat16 inputs?"""
  return not is_tpu() or tpu_generation() >= 4
