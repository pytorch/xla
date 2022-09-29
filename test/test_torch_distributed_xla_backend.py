import os
import re
import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_env_vars as xenv
import torch_xla.distributed.xla_backend
import unittest

from contextlib import contextmanager
from datetime import timedelta

# We set the following env vars to create a fake env to exercise the code under
# test. We do not aim to test device specific behaviors.
os.environ[xenv.DEVICE_MAP] = (
    'CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0')
os.environ[xenv.WORKERS] = 'localservice:0;grpc://localhost:51011'
os.environ[xenv.WORLD_SIZE] = '1'
os.environ[xenv.ORDINAL] = '0'


def get_process_group_xla(rank, size):
  pg_xla_creator = dist.Backend._plugins[dist.Backend.XLA].creator_fn
  pg_xla = pg_xla_creator(
      prefix_store=None, rank=rank, size=size, timeout=timedelta(minutes=1))
  return pg_xla


def hlo_matches(hlo, expected_pattern, match_times=1):
  matches = re.findall(expected_pattern, hlo)
  assert len(list(matches)) == match_times, hlo


def set_world_size(size):
  world_pg = dist.group.WORLD
  assert isinstance(world_pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
  world_pg.size = lambda: size


def set_world_rank(rank):
  world_pg = dist.group.WORLD
  assert isinstance(world_pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
  world_pg.rank = lambda: rank


def get_world_size():
  return dist.group.WORLD.size()


@contextmanager
def new_group_barrier_disabled():
  orig_barrier_fn = torch.distributed.distributed_c10d._store_based_barrier
  torch.distributed.distributed_c10d._store_based_barrier = lambda x, y, z: None
  try:
    yield
  finally:
    torch.distributed.distributed_c10d._store_based_barrier = orig_barrier_fn


@contextmanager
def xm_cc_op_intercepted(cc_op):
  orig_xm_cc_op = getattr(xm, cc_op)

  def build_collective_context(groups):
    cctx = xm.CollectiveContext()
    # Here we need to manipulate cctx to force xm.all_reduce to generate
    # the collective communication ops in HLO.
    # We may need to .requires_interhost_reduce in future.
    cctx.requires_intercore_reduce = True
    cctx.intercore_group = groups
    return cctx

  def patched_cc_op(*args, **kwargs):
    kwargs['cctx'] = build_collective_context(kwargs['groups'])
    return orig_xm_cc_op(*args, **kwargs)

  setattr(xm, cc_op, patched_cc_op)
  try:
    yield
  finally:
    setattr(xm, cc_op, orig_xm_cc_op)


class XlaBackendTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    dist.init_process_group(
        'xla', rank=0, world_size=1, init_method='tcp://localhost:6789')

  def test_xla_backend_exists(self):
    # torch_xla.distributed._register_xla_backend() should have been
    # automatically called.
    pg_xla_creator = dist.Backend.XLA
    assert pg_xla_creator is not None

  def test_process_group_creation(self):
    pg_xla = get_process_group_xla(rank=1, size=2)
    assert pg_xla is not None
    assert pg_xla.rank() == 1
    assert pg_xla.size() == 2

  def test_allreduce(self):
    device = xm.xla_device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    pg_xla = get_process_group_xla(rank=511, size=1024)
    opts = dist.AllreduceOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    all_reduce_pattern = r'%all\-reduce\.\d+ = .+ all\-reduce\('
    with xm_cc_op_intercepted('all_reduce'):
      pg_xla.allreduce([tensor], opts)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([tensor])
    hlo_matches(hlo, all_reduce_pattern)
    # purge all computations attached the device.
    xm.mark_step()

  def test_allreduce_with_mesh(self):
    device = xm.xla_device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()

    set_world_size(6)
    ranks = [2, 3]
    world_rank = 3
    set_world_rank(world_rank)
    with new_group_barrier_disabled():
      new_pg = dist.new_group(ranks=ranks)
    opts = dist.AllreduceOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    all_reduce_pattern = (r'%all\-reduce\.\d+ = .+ all\-reduce\(.+\), .*'
                          r'replica_groups=\{\{0,1\},\{2,3\},\{4,5\}\}')
    with xm_cc_op_intercepted('all_reduce'):
      new_pg.allreduce([tensor], opts)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([tensor])
    hlo_matches(hlo, all_reduce_pattern)
    # purge all computations attached the device.
    xm.mark_step()

  def test_allgather(self):
    device = xm.xla_device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    pg_xla = get_process_group_xla(rank=3, size=8)
    output_tensors = [torch.zeros_like(tensor)] * 8
    all_gather_pattern = r'%all\-gather\.\d+ = .+ all\-gather\('
    pg_xla.allgather([output_tensors], [tensor])
    hlo = torch_xla._XLAC._get_xla_tensors_hlo(output_tensors)
    hlo_matches(hlo, all_gather_pattern)
    # purge all computations attached the device.
    xm.mark_step()

  def test_broadcast(self):
    device = xm.xla_device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    pg_xla = get_process_group_xla(rank=0, size=8)
    opts = dist.BroadcastOptions()
    opts.rootRank = 0
    opts.rootTensor = 0

    # Sync value of `tensor` to remove constants 1 and 0 from graph
    xm.mark_step()

    # xla doesn't have broadcast. We use all_reduce to implement broadcast.
    all_reduce_pattern = r'%all\-reduce\.\d+ = .+ all\-reduce\('
    with xm_cc_op_intercepted('all_reduce'):
      pg_xla.broadcast([tensor], opts)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([tensor])
    hlo_matches(hlo, all_reduce_pattern)

    assert 'constant' not in hlo, hlo

    # purge all computations attached the device.
    xm.mark_step()

  # Needed for ZeRO stage 1
  def test_reduce_scatter(self):
    device = xm.xla_device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    input_list = [tensor]
    output = torch.zeros_like(tensor)
    pg_xla = get_process_group_xla(rank=0, size=len(input_list))
    opts = dist.ReduceScatterOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    reduce_scatter_pattern = r'%reduce\-scatter\.\d+ = .+ reduce\-scatter\('
    pg_xla.reduce_scatter([output], [input_list], opts)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([output])
    hlo_matches(hlo, reduce_scatter_pattern)
    # purge all computations attached the device.
    xm.mark_step()

  def test_send(self):
    device = xm.xla_device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    input_list = [tensor]
    set_world_size(6)
    ranks = [0, 3]
    world_rank = 0
    set_world_rank(world_rank)

    torch_xla.distributed.xla_backend.ProcessGroupXla.make_send_channel_id = (
        lambda self, dst_rank, tag: dst_rank * 2)

    with new_group_barrier_disabled():
      pg_xla = dist.new_group(ranks=ranks)

    send_pattern = r'%send\.\d+ = .+ send\(.+\), channel_id=2'
    senddone_pattern = r'%send\-done\.\d+ = .+ send\-done\(.+\), channel_id=2'
    # seeing 'Send is not implemented on CPU' means we have successfully
    # generated `send` in the HLO.
    with self.assertRaises(RuntimeError) as cm:
      pg_xla.send(input_list, 1)
      hlo = torch_xla._XLAC._get_xla_tensors_hlo(input_list)
      hlo_matches(hlo, send_pattern)
      hlo_matches(hlo, senddone_pattern)
      xm.mark_step()
    assert 'UNIMPLEMENTED: Send is not implemented on CPU.' in str(
        cm.exception), str(cm.exception)
    # reset token to clean up the mess after the RuntimeError.
    xm.set_replication(device, [])

  def test_recv(self):
    device = xm.xla_device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    output_list = [tensor]
    set_world_size(6)
    ranks = [0, 3]
    world_rank = 0
    set_world_rank(world_rank)

    torch_xla.distributed.xla_backend.ProcessGroupXla.make_recv_channel_id = (
        lambda self, src_rank, tag: src_rank * 3)

    with new_group_barrier_disabled():
      pg_xla = dist.new_group(ranks=ranks)

    recv_pattern = r'%recv\.\d+ = .+ recv\(.+\), channel_id=3'
    recvdone_pattern = r'%recv\-done\.\d+ = .+ recv\-done\(.+\), channel_id=3'
    # seeing 'recv is not implemented on CPU' means we have successfully
    # generated `recv` in the HLO.
    with self.assertRaises(RuntimeError) as cm:
      pg_xla.recv(output_list, 1)
      hlo = torch_xla._XLAC._get_xla_tensors_hlo(output_list)
      hlo_matches(hlo, recv_pattern)
      hlo_matches(hlo, recvdone_pattern)
      xm.mark_step()
    assert 'UNIMPLEMENTED: Recv is not implemented on CPU.' in str(
        cm.exception), str(cm.exception)
    # reset token to clean up the mess after the RuntimeError.
    xm.set_replication(device, [])

  def test_new_group_no_ranks(self):
    set_world_size(12)
    with new_group_barrier_disabled():
      pg = dist.new_group()
    assert isinstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
    assert pg.size() == get_world_size()

  def test_new_group_horizontal(self):
    set_world_size(12)

    ranks = [4, 5, 6, 7]
    world_rank = 5
    set_world_rank(world_rank)
    with new_group_barrier_disabled():
      pg = dist.new_group(ranks=ranks)
    assert isinstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
    assert pg.size() == len(ranks)
    assert pg.rank() == ranks.index(world_rank)
    assert pg._mesh == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    ranks = [0, 1, 2, 3]
    world_rank = 2
    set_world_rank(world_rank)
    with new_group_barrier_disabled():
      pg = dist.new_group(ranks=ranks)
    assert isinstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
    assert pg.size() == len(ranks)
    assert pg.rank() == ranks.index(world_rank)
    assert pg._mesh == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    ranks = [8, 9, 10, 11]
    world_rank = 11
    set_world_rank(world_rank)
    with new_group_barrier_disabled():
      pg = dist.new_group(ranks=ranks)
    assert isinstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
    assert pg.size() == len(ranks)
    assert pg.rank() == ranks.index(world_rank)
    assert pg._mesh == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

  def test_new_group_vertical(self):
    set_world_size(12)

    ranks = [1, 5, 9]
    world_rank = 5
    set_world_rank(world_rank)
    with new_group_barrier_disabled():
      pg = dist.new_group(ranks=ranks)
    assert isinstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
    assert pg.size() == len(ranks)
    assert pg.rank() == ranks.index(world_rank)
    assert pg._mesh == [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]

    ranks = [0, 4, 8]
    world_rank = 0
    set_world_rank(world_rank)
    with new_group_barrier_disabled():
      pg = dist.new_group(ranks=ranks)
    assert isinstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
    assert pg.size() == len(ranks)
    assert pg.rank() == ranks.index(world_rank)
    assert pg._mesh == [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]

    ranks = [3, 7, 11]
    world_rank = 11
    set_world_rank(world_rank)
    with new_group_barrier_disabled():
      pg = dist.new_group(ranks=ranks)
    assert isinstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
    assert pg.size() == len(ranks)
    assert pg.rank() == ranks.index(world_rank)
    assert pg._mesh == [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]

  def test_new_group_one_paticipant(self):
    set_world_size(12)

    ranks = [5]
    world_rank = 5
    set_world_rank(world_rank)
    with new_group_barrier_disabled():
      pg = dist.new_group(ranks=ranks)
    assert isinstance(pg,
                      torch_xla.distributed.xla_backend.ProcessGroupXla), str(
                          type(pg))
    assert pg.size() == 1
    assert pg.rank() == 0
    assert pg._mesh == [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
                        [11]]

  def test_new_group_entire_world(self):
    set_world_size(12)

    ranks = range(12)
    world_rank = 5
    set_world_rank(world_rank)
    with new_group_barrier_disabled():
      pg = dist.new_group(ranks=ranks)
    assert isinstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
    assert pg.size() == 12
    assert pg.rank() == world_rank
    assert pg._mesh == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

  def test_new_group_invalid_horizontal(self):
    set_world_size(12)

    ranks = [4, 5, 6]
    world_rank = 5
    set_world_rank(world_rank)
    with new_group_barrier_disabled():
      with self.assertRaises(ValueError):
        pg = dist.new_group(ranks=ranks)

    ranks = [0, 1, 2, 3, 4]
    world_rank = 2
    set_world_rank(world_rank)
    with new_group_barrier_disabled():
      with self.assertRaises(ValueError):
        pg = dist.new_group(ranks=ranks)

    ranks = [7, 8, 9, 10]
    world_rank = 9
    set_world_rank(world_rank)
    with new_group_barrier_disabled():
      with self.assertRaises(ValueError):
        pg = dist.new_group(ranks=ranks)

  def test_new_group_invalid_vertical(self):
    set_world_size(12)

    ranks = [1, 5]
    world_rank = 5
    set_world_rank(world_rank)
    with new_group_barrier_disabled():
      with self.assertRaises(ValueError):
        pg = dist.new_group(ranks=ranks)

    ranks = [4, 7, 10]
    world_rank = 4
    set_world_rank(world_rank)
    with new_group_barrier_disabled():
      with self.assertRaises(ValueError):
        pg = dist.new_group(ranks=ranks)

  def test_new_group_invalid_ranks(self):
    set_world_size(12)

    # unevenly distributed
    ranks = [1, 5, 10]
    world_rank = 5
    set_world_rank(world_rank)
    with new_group_barrier_disabled():
      with self.assertRaises(ValueError):
        pg = dist.new_group(ranks=ranks)

  def test_barrier(self):
    # nothing to verify. Just run it through.
    dist.barrier()

  def test_unimplemented_ops(self):
    unimplemented_ops = (
        'reduce',
        'allgather_coalesced',
        'allreduce_coalesced',
        'alltoall',
        'alltoall_base',
        'gather',
        'scatter',
        'recv_anysource',
        'monitored_barrier',
    )
    device = xm.xla_device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    pg_xla = get_process_group_xla(rank=0, size=8)
    for op in unimplemented_ops:
      with self.assertRaises(NotImplementedError):
        getattr(pg_xla, op)(tensor)


if __name__ == '__main__':
  skipping_msg = ('Skipping XLA backend unit test as this test doesn\'t '
                  'exercise %s-specific behaviors.')
  if xenv.TPU_CONFIG in os.environ:
    print(skipping_msg % 'TPU')
  elif xenv.GPU_NUM_DEVICES in os.environ:
    print(skipping_msg % 'GPU')
  else:
    unittest.main()
