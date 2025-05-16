"""Try to apply every collective in torch.distributed,
both in eager and compiled mode, and report the resulting
error for those that fail."""

import argparse
import traceback
import warnings

import torch
import torch.distributed as dist
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm


def _run_distributed_operation(
    collective_op_fn_factory, rank: int, use_dynamo: bool, input_size: int,
    verbose: bool
    # collective_op_fn_factory is a function that takes (device, input_size, rank)
    # and returns the actual callable (which itself takes input_tensor).
):
  device = xm.xla_device()

  input_tensor_val = float(xr.global_ordinal())
  input_tensor = torch.full((input_size,),
                            input_tensor_val,
                            dtype=torch.bfloat16,
                            device=device)
  if verbose:
    print(f"rank {rank} input = {input_tensor}")

  actual_callable = collective_op_fn_factory(device, input_size, rank)

  # Compile with OpenXLA backend if use_dynamo is True
  if use_dynamo:
    compiled_fn = torch.compile(
        actual_callable, backend='openxla', fullgraph=True)
    f_to_execute = compiled_fn
  else:
    f_to_execute = actual_callable

  output = f_to_execute(input_tensor)
  torch_xla.sync()

  if verbose:
    print(f"rank {rank} output = {output}")
  return output


# --- Specific Operation Factories ---
# Each factory returns a callable function that performs the collective.
# The callable will take the input_tensor as its argument.
# Output buffers are created within the factory's scope and captured by the callable.


def _broadcast_factory(device: torch.device, input_size: int, rank: int):

  def callable_fn(input_t: torch.Tensor):
    dist.broadcast(input_t, src=0)
    return input_t

  return callable_fn


def _broadcast_object_list_factory(device: torch.device, input_size: int,
                                   rank: int):

  def callable_fn(input_t: torch.Tensor):
    dist.broadcast_object_list([input_t], src=0)
    return input_t

  return callable_fn


def _all_reduce_factory(device: torch.device, input_size: int, rank: int):
  """Factory for the all_reduce operation callable."""

  def callable_fn(input_t: torch.Tensor):
    dist.all_reduce(input_t, dist.ReduceOp.SUM)
    return input_t  # all_reduce modifies input_t in-place

  return callable_fn


def _reduce_factory(device: torch.device, input_size: int, rank: int):

  def callable_fn(input_t: torch.Tensor):
    dist.reduce(input_t, dst=0, op=dist.ReduceOp.SUM)
    return input_t

  return callable_fn


def _gather_factory(device: torch.device, input_size: int, rank: int):
  """Factory for the gather operation callable."""
  output_tensors_for_capture = [
      torch.zeros((input_size,), dtype=torch.bfloat16, device=device)
      for _ in range(xr.world_size())
  ]

  def callable_fn(input_t: torch.Tensor):
    dist.gather(
        input_t,
        gather_list=output_tensors_for_capture if rank == 0 else None,
        dst=0)
    return output_tensors_for_capture  # Meaningful content only on rank 0

  return callable_fn


def _all_gather_into_tensor_factory(device: torch.device, input_size: int,
                                    rank: int):
  """Factory for the all_gather_into_tensor operation callable."""
  # The output_tensor is pre-allocated and captured by the callable.
  # Its shape is (world_size, input_size_per_rank).
  output_tensor_agg = torch.zeros((xr.world_size(), input_size),
                                  dtype=torch.bfloat16,
                                  device=device)

  def callable_fn(input_t: torch.Tensor):
    dist.all_gather_into_tensor(output_tensor_agg, input_t)
    return output_tensor_agg  # output_tensor_agg is modified in-place

  return callable_fn


def _all_gather_factory(device: torch.device, input_size: int, rank: int):
  """Factory for the all_gather operation callable."""
  # List of output tensors, pre-allocated and captured.
  output_tensors_list = [
      torch.zeros((input_size,), dtype=torch.bfloat16, device=device)
      for _ in range(xr.world_size())
  ]

  def callable_fn(input_t: torch.Tensor):
    dist.all_gather(tensor_list=output_tensors_list, tensor=input_t)
    return output_tensors_list  # Elements of the list are modified

  return callable_fn


def _scatter_factory(device: torch.device, input_size: int, rank: int):
  scatter_list = [
      torch.full((input_size,),
                 fill_value=i,
                 dtype=torch.bfloat16,
                 device=device) for i in range(xr.world_size())
  ]

  def callable_fn(input_t: torch.Tensor):
    dist.scatter(
        input_t, scatter_list=scatter_list if rank == 0 else None, src=0)
    return input_t

  return callable_fn


def _reduce_scatter_factory(device: torch.device, input_size: int, rank: int):
  scatter_list = [
      torch.full((input_size,),
                 fill_value=i,
                 dtype=torch.bfloat16,
                 device=device) for i in range(xr.world_size())
  ]

  def callable_fn(input_t: torch.Tensor):
    dist.reduce_scatter(input_t, input_list=scatter_list, op=dist.ReduceOp.SUM)
    return input_t

  return callable_fn


def _reduce_scatter_tensor_factory(device: torch.device, input_size: int,
                                   rank: int):
  scatter_tensor = torch.stack([
      torch.full((input_size,),
                 fill_value=i,
                 dtype=torch.bfloat16,
                 device=device) for i in range(xr.world_size())
  ])

  # This also works if the tensors to scatter are concatenated, in which case we don't need
  # to reshape input_t to be 2 dimensional.
  def callable_fn(input_t: torch.Tensor):
    input_t = input_t.reshape(1, -1)
    dist.reduce_scatter_tensor(
        input_t, input=scatter_tensor, op=dist.ReduceOp.SUM)
    return input_t

  return callable_fn


def _all_to_all_factory(device: torch.device, input_size: int, rank: int):

  def callable_fn(input_t: torch.Tensor):
    output_tensor_list = [torch.zeros_like(input_t)]
    dist.all_to_all(output_tensor_list, [input_t])
    return output_tensor_list

  return callable_fn


def _all_to_all_single_factory(device: torch.device, input_size: int,
                               rank: int):
  output_tensor = torch.zeros((input_size,),
                              device=device,
                              dtype=torch.bfloat16)

  def callable_fn(input_t: torch.Tensor):
    dist.all_to_all_single(output_tensor, input_t)
    return output_tensor

  return callable_fn


def _send_recv_factory(device: torch.device, input_size: int, rank: int):

  def callable_fn(input_t: torch.Tensor):
    output_t = torch.zeros_like(input_t)
    dist.send(input_t, dst=(rank + 1) % xr.world_size())
    dist.recv(output_t, src=(rank - 1) % xr.world_size())
    return output_t

  return callable_fn


def main(rank: int, verbose: bool = False, input_size: int = 8):
  if not verbose:
    warnings.filterwarnings("ignore")
  dist.init_process_group("xla", init_method='xla://')

  ops_to_test = [
      ("broadcast", _broadcast_factory),
      ("broadcast object list", _broadcast_object_list_factory),
      ("all reduce", _all_reduce_factory),
      ("reduce", _reduce_factory),
      ("all gather", _all_gather_factory),
      ("gather", _gather_factory),
      ("all gather into tensor", _all_gather_into_tensor_factory),
      ("scatter", _scatter_factory),
      ("reduce scatter", _reduce_scatter_factory),
      ("reduce scatter tensor", _reduce_scatter_tensor_factory),
      ("all to all", _all_to_all_factory),
      ("all to all single", _all_to_all_single_factory),
      ("send receive", _send_recv_factory),
  ]

  for key, factory in ops_to_test:
    for use_dynamo in [False, True]:
      if use_dynamo:
        descriptor_str = f"{key.upper()}, dynamo"
      else:
        descriptor_str = f"{key.upper()}, eager"
      if rank == 0:
        print(descriptor_str)
      torch_xla.sync(wait=True)
      result_str = "SUCCESS"
      try:
        _run_distributed_operation(
            collective_op_fn_factory=factory,
            rank=rank,
            use_dynamo=use_dynamo,
            input_size=input_size,
            verbose=verbose,
        )
      except Exception as e:
        result_str = f"FAILED with exception {type(e).__name__}"
        if verbose:
          result_str += f"\t{e}\n{traceback.format_exc()}"
      torch_xla.sync(wait=True)
      if rank == 0:
        print(result_str)

  dist.destroy_process_group()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Test collective operations")
  parser.add_argument(
      "-v",
      "--verbose",
      action="store_true",
      help="Print inputs, outputs, and full error messages.")
  parser.add_argument(
      "--input_size",
      type=int,
      default=8,
      help="Size of the initial tensor on each device")
  args = parser.parse_args()
  torch_xla.launch(
      main, args=(args.verbose, args.input_size), debug_single_process=False)
