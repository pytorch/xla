import copy
import shutil
import os
import re
import numpy as np
import torch
from torch import nn
import torch_xla
from torch_xla.core import xla_model as xm
from torch_xla.core import dynamo_bridge
from torch_xla.debug import metrics
import tensorflow as tf
import torchvision
import torch._dynamo as torchdynamo

from typing import Tuple, Type, Callable

import sys


def _get_numpy_dtype(dtype):
  return {
      torch.double: np.double,
      torch.float32: np.float32,
      torch.half: np.half,
      torch.long: np.int64,
      torch.int32: np.int32,
      torch.int16: np.int16,
      torch.bool: np.bool_,
  }.get(dtype)


class SHLOModel:

  def __init__(self, model_bytecode, pos_to_orig_pos, pos_to_param,
               output_shape, output_dtype):
    self.model_bytecode = model_bytecode
    self.pos_to_orig_pos = pos_to_orig_pos
    self.pos_to_param = pos_to_param
    self._total_number = len(pos_to_orig_pos) + len(pos_to_param)

    self.tout = (_get_numpy_dtype(output_dtype),)
    self.sout = (tuple(output_shape),)

  def __call__(self, *args):
    # TODO(qihqi): remove this
    from tensorflow.compiler.tf2xla.python import xla as tfxla
    call_args = []
    for i in range(self._total_number):
      if i in self.pos_to_orig_pos:
        call_args.append(args[self.pos_to_orig_pos[i]])
      else:
        call_args.append(self.pos_to_param[i])

    return tfxla.call_module(
        tuple(call_args),
        version=5,
        Tout=self.tout,  # dtype information
        Sout=self.sout,  # Shape information
        function_list=[],
        platforms=('CPU',),
        module=self.model_bytecode,
    )[0]


def export_torch_model(model: torch.nn.Module,
                       sample_inputs: Tuple[torch.Tensor]):
  """Convert model into a callable backed by StableHLO.

  Args:
      model: torch.nn.Module - a pytorch model
      sample_inputs: Tuple[torch.Tensor] - The input to this model
      output_shape: Tuple[int] - Shape of the output
      output_dtype: dtype - numpy dtype for the output

  This function will return a callable backed by StableHLO such that,

  model(*sample_inputs) ==  export_torch_model(model, sample_inputs)(*sample_inputs)
  (up to numerics)

  In other words, returned callable have the same calling convention of the input model, and
  on the sample input, or inputs sufficiently similar* to sample input,
  it is will to return same result as the original model.

  * sufficiently similar input because this function will use tracing to extract the model operations
    so it might specialize on the shapes of the sample input.

  For now, model has to only take Tensors as input and has to return 1 tensor as output.

  """

  # Materialize the computation before this call to make sure that StableHLO
  # captured below only reflect the model passed in
  xm.mark_step()
  metrics.clear_counters()
  model = copy.deepcopy(model)
  sample_inputs = copy.deepcopy(sample_inputs)

  device = xm.xla_device()
  model = model.to(device=device)
  sample_inputs_lazy = tuple(
      x.to(device=device) if isinstance(x, torch.Tensor) else x
      for x in sample_inputs)
  input_ids = {
      torch_xla._XLAC._xla_get_tensor_id(tensor): i
      for i, tensor in enumerate(sample_inputs_lazy)
      if isinstance(tensor, torch.Tensor)
  }
  output = model(*sample_inputs_lazy)
  if fallback_ops := dynamo_bridge.get_fallback_ops():
    message = "This model contains ops not capturable by Pytorch/XLA: {}".format(
        "\n".join(fallback_ops))
    raise RuntimeError(message)
  stablehlo_bytecode = xm.get_stablehlo_bytecode([output])

  (
      graph_input_tensor_ids,
      graph_input_xla_values,
  ) = torch_xla._XLAC._get_tensors_xla_device_data_node([output])

  pos_to_orig_pos = {}
  pos_to_param = {}
  for hlo_input_pos, tensor_id in enumerate(graph_input_tensor_ids):
    if tensor_id in input_ids:  # this is input
      pos_to_orig_pos[hlo_input_pos] = input_ids[tensor_id]
    else:
      pos_to_param[hlo_input_pos] = graph_input_xla_values[
          hlo_input_pos].detach().cpu().numpy()

  stablehlo_model = SHLOModel(stablehlo_bytecode, pos_to_orig_pos, pos_to_param,
                              output.shape, output.dtype)
  # Remove all of the pending IR from all live tensors. The assumptions are
  # 1. With the  `xm.mark_step` in the beginning of this call, every XLATensor
  # should be materialized
  # 2. All of the pending IRs are result of the one inference
  # should be removed to avoid extra computation executed and in place updates op
  # mistakenlly update the input tensors.
  torch_xla._XLAC._clear_pending_irs(str(xm.xla_device()))
  return stablehlo_model
