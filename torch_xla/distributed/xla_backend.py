import distutils.util
import os
import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import logging
from torch._C._distributed_c10d import (
    ProcessGroup,
    Work,
)


def _create_xla_process_group(prefix_store, rank, size, timeout):
  return ProcessGroupXla(prefix_store, rank, size, timeout)


def _register_xla_backend():
  dist.Backend.register_backend('xla', _create_xla_process_group)


print('_register_xla_backend', flush=True)
_register_xla_backend()


class ProcessGroupXla(ProcessGroup):
  '''ProcessGroup for XLA devices. See ProcessGroup for doc.

    https://github.com/pytorch/pytorch/blob/074c77601114c66df346c5465667ce8cfaa6d22c/torch/csrc/distributed/c10d/ProcessGroup.hpp#L79
    https://github.com/pytorch/pytorch/blob/074c77601114c66df346c5465667ce8cfaa6d22c/torch/csrc/distributed/c10d/init.cpp#L937
    '''

  def __init__(self, prefix_store, rank, size, timeout):
    super().__init__(rank, size)
    self.prefix_store = prefix_store  # reserved for future use.
    self.timeout = timeout
    self._mesh = []

  def _get_reduce_type(self, reduce_op):
    if reduce_op == dist.ReduceOp.SUM:
      return xm.REDUCE_SUM
    elif reduce_op == dist.ReduceOp.PRODUCT:
      return xm.REDUCE_MUL
    elif reduce_op == dist.ReduceOp.BAND:
      return xm.REDUCE_AND
    elif reduce_op == dist.ReduceOp.BOR:
      return xm.REDUCE_OR
    elif reduce_op == dist.ReduceOp.MIN:
      return xm.REDUCE_MIN
    elif reduce_op == dist.ReduceOp.MAX:
      return xm.REDUCE_MAX
    elif reduce_op == dist.ReduceOp.BXOR:
      raise NotImplementedError(f'reduce op {reduce_op}')
    else:
      raise ValueError(f'Invalid reduce op {reduce_op}')

  def allreduce(self, tensors, all_reduce_options):
    reduce_type = self._get_reduce_type(all_reduce_options.reduceOp)

    # TODO(junminh): implement all_reduce_options.timeout.
    xm.all_reduce(reduce_type, tensors, groups=self._mesh)
    return WorkXla(tensors)

  def allgather(self, output_tensors_list, input_tensors):
    for input_tensor, output_tensors in zip(input_tensors, output_tensors_list):
      result = xm.all_gather(input_tensor, groups=self._mesh)
      for i, slice in enumerate(torch.split(result, input_tensor.shape[0])):
        output_tensors[i].set_(slice)

    return WorkXla([t for sublist in output_tensors_list for t in sublist])

  # Call site:
  # https://github.com/pytorch/pytorch/blob/70f57bcb1e45d21532bdb1c44d3aab018d1cbe88/torch/distributed/distributed_c10d.py#L1138
  def broadcast(self, tensors, opts):
    root_tensor = tensors[opts.rootTensor]
    root_rank = opts.rootRank
    if root_rank != self.rank():
      with torch.no_grad():
        root_tensor.zero_()
    xm.all_reduce(xm.REDUCE_SUM, [root_tensor], groups=self._mesh)

    return WorkXla([root_tensor])

  # Call site:
  # https://github.com/pytorch/pytorch/blob/70f57bcb1e45d21532bdb1c44d3aab018d1cbe88/torch/distributed/distributed_c10d.py#L2355
  def reduce_scatter(self, output_tensors, input_tensors_list, opts):
    for input_tensors, output_tensor in zip(input_tensors_list, output_tensors):
      # Ensure all inputs have the same shape.
      last_shape = input_tensors[0].shape
      for i, t in enumerate(input_tensors[1:]):
        if last_shape != t.shape:
          raise ValueError(f"Input {i+1}'s shape is different from input {i}: "
                           f"{t.shape} vs {last_shape}")
        last_shape = t.shape
      input_tensor = torch.cat(input_tensors)
      reduce_type = self._get_reduce_type(opts.reduceOp)
      groups = self._mesh
      shard_count = len(groups[0]) if groups else self.size()
      result = xm.reduce_scatter(
          reduce_type,
          input_tensor,
          scatter_dim=0,
          shard_count=shard_count,
          scale=1,
          groups=groups)

      output_tensor.set_(result)

    return WorkXla(output_tensors)

  # Call site:
  # https://github.com/pytorch/pytorch/blob/70f57bcb1e45d21532bdb1c44d3aab018d1cbe88/torch/distributed/distributed_c10d.py#L2683
  def barrier(self, opts):
    return WorkXla()

  # Call site:
  # https://github.com/pytorch/pytorch/blob/70f57bcb1e45d21532bdb1c44d3aab018d1cbe88/torch/distributed/distributed_c10d.py#L1417
  # We will not likely need `reduce`. see
  # https://quip-amazon.com/dCxeAuZjrdtN/DeepSpeed-on-torchxla#BBU9CAhzaWi
  def reduce(self, *args):
    raise NotImplementedError

  def allgather_coalesced(self, *args):
    raise NotImplementedError

  def allreduce_coalesced(self, *args):
    raise NotImplementedError

  def alltoall(self, *args):
    raise NotImplementedError

  def alltoall_base(self, *args):
    raise NotImplementedError

  def gather(self, *args):
    raise NotImplementedError

  def scatter(self, *args):
    raise NotImplementedError

  def send(self, *args):
    raise NotImplementedError

  def recv(self, *args):
    raise NotImplementedError

  def recv_anysource(self, *args):
    raise NotImplementedError

  def monitored_barrier(self, *args):
    raise NotImplementedError

  def Options(self, *args):
    raise NotImplementedError


class WorkXla(Work):

  def __init__(self, cc_tensors=None):
    '''
        Args:
            cc_tensors (List[torch.Tensor]): List of `torch.Tensor`s that
            have collective communication ops pending.
            For each Tensor `t` in the list, `t.device` must be an `xla`
            device.
        '''
    super().__init__()
    self.cc_tensors = cc_tensors

  def wait(self):
    if self.cc_tensors is not None:
      if distutils.util.strtobool(
          os.environ.get('XLA_BACKEND_BLOCKING_CC_OPS', 'False')):
        logging.info("XLA Backend: Waiting for tensor CC op...")
        torch_xla._XLAC._xla_sync_multi(self.cc_tensors, devices=[], wait=True)
      else:
        logging.info("XLA Backend: Skipping tensor CC op wait.")
    else:
      if distutils.util.strtobool(
          os.environ.get('XLA_BACKEND_BLOCKING_BARRIER', 'False')):
        logging.info("XLA Backend: Non-tensor wait...")
        torch_xla._XLAC._xla_step_marker(
            torch_xla._XLAC._xla_get_default_device(), devices=[], wait=True)
      else:
        logging.info("XLA Backend: Skipping non-tensor wait.")


# -------------------------------------
# Override torch.distributed.new_group.
# -------------------------------------
_orig_new_group_fn = dist.new_group


def infer_mesh(slice_ranks, world_size):
  '''Infers a rectangular mesh topology from a slice in the mesh.

    Example, given world size 12 and a slice like the following:
        [1, 5, 9]
    this function infers the following mesh:
        [[0, 4, 8],
         [1, 5, 9],
         [2, 6, 10],
         [3, 7, 11]]

    We only support ractangular meshes.
    '''
  slice_len = len(slice_ranks)
  if world_size % slice_len != 0:
    raise ValueError(
        'Given slice length doesn\'t equal to a side of a rectangle. '
        f'World size: {world_size}, slice length: {slice_len}')

  # ensure the ranks are a correct range.
  start = slice_ranks[0]
  step = slice_ranks[1] - slice_ranks[0]
  stop = start + step * slice_len
  expected_ranks = list(range(start, stop, step))
  if slice_ranks != expected_ranks:
    raise ValueError(
        f'Given slice isn\'t a range with evenly distributed steps: {slice_ranks}'
    )

  num_slices = int(world_size / slice_len)
  world = list(range(world_size))
  mesh = []
  if step == 1:
    # Horizontal case.
    if start % slice_len != 0 or start >= world_size:
      raise ValueError('Given horizontal slice doesn\'t have a correct start: '
                       f'World size: {world_size}, slice length: {slice_len}, '
                       f'slice: {slice_ranks}')
    for i in range(num_slices):
      slice_start = i * slice_len
      slice_stop = (i + 1) * slice_len
      slice_idx = slice(slice_start, slice_stop, step)
      mesh.append(world[slice_idx])
  else:
    # Vertical case.
    if start >= num_slices:
      raise ValueError('Given vertical slice doesn\'t have a correct start: '
                       f'World size: {world_size}, slice length: {slice_len}, '
                       f'slice: {slice_ranks}')
    if step != num_slices:
      raise ValueError('Given vertical slice doesn\'t have a correct step: '
                       f'World size: {world_size}, slice length: {slice_len}, '
                       f'slice: {slice_ranks}')
    for i in range(num_slices):
      slice_start = i
      slice_stop = i + slice_len * step
      slice_idx = slice(slice_start, slice_stop, step)
      mesh.append(world[slice_idx])
  assert slice_ranks in mesh

  return mesh


def new_xla_process_group(ranks=None,
                          timeout=dist.default_pg_timeout,
                          backend=None):
  pg = _orig_new_group_fn(ranks, timeout, backend)
  if isinstance(pg, ProcessGroupXla) and ranks is not None:
    world_pg = dist.group.WORLD
    if not isinstance(world_pg, ProcessGroupXla):
      raise RuntimeError('xla backend requires the default ProcessGroup to be '
                         'a ProcessGroupXla')

    if isinstance(ranks, range):
      ranks = list(ranks)

    if ranks == list(range(world_pg.size())):
      pg._mesh = [ranks]
    elif len(ranks) == 1:
      if ranks[0] not in range(world_pg.size()):
        raise ValueError('Given ranks is out of range: '
                         f'World size: {world_pg.size()}, ranks: {ranks}')
      pg._mesh = [[r] for r in range(world_pg.size())]
    elif len(ranks) < world_pg.size() and len(ranks) > 1:
      pg._mesh = infer_mesh(ranks, world_pg.size())
    else:
      logging.warn(
          f'Can\'t infer process group mesh from given ranks "{str(ranks)}". '
          'The process group will use the entire world as its collective comm group.'
      )

  return pg


dist.new_group = new_xla_process_group
# -------------------------------------------
# End overriding torch.distributed.new_group.
# -------------------------------------------
