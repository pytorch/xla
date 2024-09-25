import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torch_xla._internal import rendezvous
import logging
import os
from torch._C._distributed_c10d import ProcessGroup


def _create_xla_process_group(prefix_store, rank, size, timeout):
  assert not xr.is_spmd(
  ), "XLA backend is not supported with SPMD. Please use a CPU process group instead."
  return ProcessGroupXla(prefix_store, rank, size, timeout)


def _register_xla_backend():
  dist.Backend.register_backend('xla', _create_xla_process_group, devices='xla')


_register_xla_backend()

dist.register_rendezvous_handler('xla', rendezvous.pjrt_rendezvous_handler)


def _ret_work(ret):
  fut = torch.futures.Future()
  fut.set_result(ret)
  return torch._C._distributed_c10d._create_work_from_future(fut)


class ProcessGroupXla(ProcessGroup):
  '''ProcessGroup for XLA devices. See ProcessGroup for doc.

    Here we are implementing only a Python subclass. For implementing a
    C++/Python extension, see
    https://pytorch.org/tutorials/intermediate/process_group_cpp_extension_tutorial.html.
    '''

  def __init__(self, prefix_store, rank, size, timeout):
    super().__init__(rank, size)
    self.prefix_store = prefix_store  # reserved for future use.
    self.timeout = timeout
    self._mesh = []

  def getBackendName(self):
    return 'xla'

  # pytorch's process group is unable to retrive the group size from python level. It should
  # already been support in C++ level: https://github.com/pytorch/pytorch/blob/7b1988f9222f3dec5cc2012afce84218199748ae/torch/csrc/distributed/c10d/ProcessGroup.cpp#L148-L152
  # For now we manually set the group name property as a temporary solution.
  def _set_group_name(self, name: str) -> None:
    self._group_name = name

  @property
  def group_name(self):
    return self._group_name

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

    # TODO(hjm-aws): implement all_reduce_options.timeout.
    xm.all_reduce(reduce_type, tensors, groups=self._mesh, pin_layout=False)
    return _ret_work(tensors)

  # method for dist.all_gather_into_tensor under eager mode.
  def _allgather_base(self, output_tensor, input_tensor, opts):
    return self.allgather(output_tensor, input_tensor, opts)

  def allgather(self, output_tensors_list, input_tensors, opts=None):
    for input_tensor, output_tensors in zip(input_tensors, output_tensors_list):
      is_scalar = (input_tensor.dim() == 0)
      if is_scalar:
        input_tensor = torch.reshape(input_tensor, (1,))
      result = xm.all_gather(input_tensor, groups=self._mesh, pin_layout=False)
      for i, slice in enumerate(torch.split(result, input_tensor.shape[0])):
        with torch.no_grad():
          output_tensors[i].copy_(
              slice if not is_scalar else torch.reshape(slice, ()))

    return _ret_work([t for sublist in output_tensors_list for t in sublist])

  def allgather_coalesced(self, output_tensors_list, input_tensors, opts=None):
    results = xm.all_gather(input_tensors, groups=self._mesh, pin_layout=False)
    for i, result in enumerate(results):
      for j, slice in enumerate(torch.split(result, input_tensors[i].shape[0])):
        output_tensors_list[i][j].copy_(slice)

    return _ret_work([t for sublist in output_tensors_list for t in sublist])

  # Call site:
  # https://github.com/pytorch/pytorch/blob/release/1.10/torch/distributed/distributed_c10d.py#L1129
  def broadcast(self, tensors, opts):
    root_tensor = tensors[opts.rootTensor]
    xm.collective_broadcast([root_tensor],
                            opts.rootRank,
                            groups=self._mesh,
                            pin_layout=False)

    return _ret_work([root_tensor])

  # Call site:
  # https://github.com/pytorch/pytorch/blob/release/1.10/torch/distributed/distributed_c10d.py#L2355
  def reduce_scatter(self, output_tensors, input_tensors_list, opts):
    for input_tensors, output_tensor in zip(input_tensors_list, output_tensors):
      # Ensure all inputs have the same shape.
      first_shape = input_tensors[0].shape
      for i, t in enumerate(input_tensors[1:]):
        if first_shape != t.shape:
          raise ValueError(f"Input {i+1}'s shape is different from input 0: "
                           f"{t.shape} vs {first_shape}")
      input_tensor = torch.cat(input_tensors)
      reduce_type = self._get_reduce_type(opts.reduceOp)
      groups = self._mesh
      shard_count = len(groups[0]) if groups else self.size()
      xm.reduce_scatter(
          reduce_type,
          input_tensor,
          scatter_dim=0,
          shard_count=shard_count,
          scale=1,
          groups=groups,
          output=output_tensor,
          pin_layout=False)

    return _ret_work(output_tensors)

  def reduce_scatter_coalesced(self, output_tensors, input_tensors_list, opts):
    input_tensor_list = []
    for input_tensors in input_tensors_list:
      # Ensure all inputs have the same shape.
      first_shape = input_tensors[0].shape
      for i, t in enumerate(input_tensors[1:]):
        if first_shape != t.shape:
          raise ValueError(f"Input {i+1}'s shape is different from input 0: "
                           f"{t.shape} vs {first_shape}")
      input_tensor = torch.cat(input_tensors)
      input_tensor_list.append(input_tensor)

    reduce_type = self._get_reduce_type(opts.reduceOp)
    groups = self._mesh
    shard_count = len(groups[0]) if groups else self.size()
    xm.reduce_scatter(
        reduce_type,
        input_tensor_list,
        scatter_dim=0,
        shard_count=shard_count,
        scale=1,
        groups=groups,
        output=output_tensors,
        pin_layout=False)

    return _ret_work(output_tensors)

  # call site https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/torch/distributed/distributed_c10d.py#L3856
  def _reduce_scatter_base(self, output_tensor, input_tensor, opts):
    """
    Reduces, then scatters a flattened tensor to all processes in a group.

    Args:
        output (Tensor): Output tensor.
        input (Tensor): Input tensor that is of size output tensor size times world size
        opts: distributed reduce op (ReduceOp).

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.
    """
    reduce_type = self._get_reduce_type(opts.reduceOp)
    groups = self._mesh
    shard_count = len(groups[0]) if groups else self.size()
    xm.reduce_scatter(
        reduce_type,
        input_tensor,
        scatter_dim=0,
        shard_count=shard_count,
        scale=1.0,
        groups=groups,
        output=output_tensor,
        pin_layout=False)
    return _ret_work(output_tensor)

  # Call site:
  # https://github.com/pytorch/pytorch/blob/70f57bcb1e45d21532bdb1c44d3aab018d1cbe88/torch/distributed/distributed_c10d.py#L2683
  def barrier(self, opts):
    return _ret_work([])

  # Call site:
  # https://github.com/pytorch/pytorch/blob/70f57bcb1e45d21532bdb1c44d3aab018d1cbe88/torch/distributed/distributed_c10d.py#L1417
  # `reduce` is not needed by DeepSpeed for now.
  def reduce(self, *args):
    raise NotImplementedError

  def allreduce_coalesced(self, *args):
    raise NotImplementedError

  def alltoall(self, *args):
    raise NotImplementedError

  # handle the nondynamo path when call torch.distributed.all_to_all_single
  # call from https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/torch/distributed/distributed_c10d.py#L3996
  # Note for pytorch, the split/concat dimension is always 0, while for XLA alltoall,
  # we can't specify different split sizes.
  def alltoall_base(self, output, input, output_split_sizes, input_split_sizes,
                    opts):
    assert (output_split_sizes is None or len(output_split_sizes) == 0) and \
           (input_split_sizes is None or len(input_split_sizes) == 0), \
           "XLA doesn't support specifying non-empty output_split_sizes and input_split_sizes"
    split_count = xr.world_size()
    result = xm.all_to_all(input, 0, 0, split_count, pin_layout=False)
    output.copy_(result)
    return _ret_work(output)

  def gather(self, *args):
    raise NotImplementedError

  def scatter(self, *args):
    raise NotImplementedError

  # Dummy channel id maker. Different backend (TPU, GPU, etc) should replace
  # the maker with their specific one. See unit test in
  # test/test_torch_distributed_xla_backend.py for an example.
  def make_send_channel_id(self, dst_rank, tag):
    raise NotImplementedError

  # Call site e.g.
  # https://github.com/pytorch/pytorch/blob/release/1.10/torch/distributed/distributed_c10d.py#L877
  def send(self, tensors, dst_rank, tag=0):
    results = []
    for t in tensors:
      channel_id = self.make_send_channel_id(dst_rank, tag)
      # The input will be returned as result.
      input_as_result = xm.send(t, channel_id)
      # Make the sent tensor depend on the token, such that the `send`
      # op can actually be built into the computation graph.
      with torch.no_grad():
        t.copy_(input_as_result)
      results.append(input_as_result)
    return _ret_work(results)

  # Dummy channel id maker. Different backend (TPU, GPU, etc) should replace
  # the maker with their specific one. See unit test in
  # test/test_torch_distributed_xla_backend.py for an example.
  def make_recv_channel_id(self, src_rank, tag):
    raise NotImplementedError

  # Call site e.g.
  # https://github.com/pytorch/pytorch/blob/release/1.10/torch/distributed/distributed_c10d.py#L913
  def recv(self, out_tensors, src_rank, tag=0):
    results = []
    for ot in out_tensors:
      channel_id = self.make_recv_channel_id(src_rank, tag)
      result = xm.recv(ot, channel_id)
      results.append(result)
    return _ret_work(results)

  def recv_anysource(self, *args):
    raise NotImplementedError

  def monitored_barrier(self, *args):
    raise NotImplementedError

  def Options(self, *args):
    raise NotImplementedError


# -------------------------------------
# Override torch.distributed.new_group.
# -------------------------------------
_orig_new_group_fn = dist.new_group


def _infer_mesh(slice_ranks, world_size):
  '''Infers a rectangular mesh topology from a slice in the mesh.

    Example, given world size 12 and a slice like the following:
        [1, 5, 9]
    this function infers the following mesh:
        [[0, 4, 8],
         [1, 5, 9],
         [2, 6, 10],
         [3, 7, 11]]

    We only support rectangular meshes.
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
                          backend=None,
                          pg_options=None):
  #this options tells xla backend to "infer" a mesh
  use_spmd = False
  #this option allows the application to pass in the mesh
  mesh_spmd = None
  if pg_options is not None and isinstance(pg_options, dict):
    if 'xla_pg_options' in pg_options:
      mesh_spmd = pg_options['xla_pg_options'].get('mesh', None)
      if mesh_spmd == None:
        use_spmd = pg_options['xla_pg_options'].get('spmd', False)

  pg = _orig_new_group_fn(
      ranks=ranks, timeout=timeout, backend=backend, pg_options=pg_options)
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
      if mesh_spmd:
        pg._mesh = mesh_spmd
      elif use_spmd:
        pg._mesh = _infer_mesh(ranks, world_pg.size())
      else:
        pg._mesh = [ranks]
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
