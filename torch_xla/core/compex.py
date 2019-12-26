from __future__ import division
from __future__ import print_function

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl


def flatten_xla_tensors(data):
  tensors = []

  def select_fn(value):
    return type(value) == torch.Tensor and xm.is_xla_tensor(value)

  def collect_fn(value):
    tensors.append(value)

  xu.for_each_instance(data, select_fn, collect_fn)
  return tensors


def run(loader,
        device,
        closure,
        closure_args=(),
        output_closure=None,
        output_closure_args=()):
  para_loader = pl.ParallelLoader(loader, [device], fixed_batch_size=True)
  device_loader = para_loader.per_device_loader(device)
  prev_hash = None
  handle_map = dict()
  steady_graph = None
  outputs = None
  for batch in device_loader:
    if output_closure is not None and outputs is not None:
      output_closure(outputs, *output_closure_args)
    if steady_graph:
      outputs = torch_xla._XLAC._xla_execute_compiled_graph(
          flatten_xla_tensors(batch), steady_graph)
    else:
      tensors = closure(batch, *closure_args)
      graph_dict = torch_xla._XLAC._xla_compile_execute_graph(
          flatten_xla_tensors(batch), tensors, str(device), [], handle_map)
      if graph_dict is None:
        raise RuntimeError('Unable to accelarate graph execution')
      chash = graph_dict['hash']
      if chash == prev_hash:
        steady_graph = graph_dict['graph']
        handle_map = None
      else:
        prev_hash = chash
        handle_map = graph_dict['handle_map']
      outputs = graph_dict['outputs']
      # Release the compile graph dictionary to make sure we do not hold two
      # copies of it while reaching stable compilations.
      graph_dict = None
    xm.mark_step_trail()
