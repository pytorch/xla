import os

import torch
import torch_xla
from torch_xla.core import xla_model as xm
from torch_xla.core.xla_model import XLA_LIB

# Enable debug info automatically when importing this file. This is necessary
# to propagate any debug info to downstream MLIR locations.
os.environ["XLA_HLO_DEBUG"] = "1"

XLA_LIB.define("write_mlir_debuginfo(Tensor x, str data) -> Tensor")


@torch.library.impl(XLA_LIB, "write_mlir_debuginfo", "XLA")
def write_mlir_debuginfo(x, data: str):
  begin_token = "<XLA_MLIR_DEBUGINFO_BEGIN>"
  end_token = "<XLA_MLIR_DEBUGINFO_END>"
  # Add the debuginfo string as the op prefix in MLIR location, surrounded
  # by begin and end tokens. The tokens and suffix op name will be removed
  # in the downstream pass PrepareXlaMlirDebuginfoPass after converting
  # HLO proto to MLIR.
  torch_xla._XLAC._set_xla_custom_op_name_prefix(
      x,
      begin_token + data + end_token,
      0,
  )
  return x


@torch.library.impl(XLA_LIB, "write_mlir_debuginfo",
                    "CompositeExplicitAutograd")
def write_mlir_debuginfo_tensor(x, data: str):
  return x


@torch.library.impl(XLA_LIB, "write_mlir_debuginfo", "Meta")
def write_mlir_debuginfo_meta(x, data: str):
  return torch.empty_like(x)
