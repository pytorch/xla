import os
import pytest
import random
import re
import string
import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_env_vars as xenv
import torch_xla.distributed.xla_backend

from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

os.environ[xenv.WORLD_SIZE] = '1'
os.environ[xenv.ORDINAL] = '0'
os.environ[
    xenv.
    DEVICE_MAP] = 'CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0'
os.environ['XLA_SAVE_TENSORS_FMT'] = 'hlo'
os.environ['XLA_BACKEND_BLOCKING_CC_OPS'] = 'True'


def get_process_group_xla(rank, size):
  pg_xla_creator = dist.Backend._plugins[dist.Backend.XLA]
  pg_xla = pg_xla_creator(
      prefix_store=None, rank=rank, size=size, timeout=timedelta(minutes=1))
  return pg_xla


def random_filename():
  return ''.join(random.choices(string.ascii_letters + string.digits, k=12))


@contextmanager
def hlo_dump_monitored(hlo_pattern, match_times=1):
  if 'XLA_SAVE_TENSORS_FILE' not in os.environ:
    os.environ['XLA_SAVE_TENSORS_FILE'] = random_filename() + '.hlo'
  try:
    yield
  finally:
    path = Path(os.environ['XLA_SAVE_TENSORS_FILE'])
    assert path.exists()
    hlo = path.read_text()
    # Note here these measures doesn't guarentee the temp file content
    # being deleted -- the C++ layer will keep flushing in the content
    # from last run. The only workaround I found is using pytest multi
    # workers so each worker only run one test.
    os.remove(os.environ['XLA_SAVE_TENSORS_FILE'])
    os.sync()
    assert not path.exists()
    matches = re.findall(hlo_pattern, hlo)
    assert len(list(matches)) == match_times, hlo


def hlo_matches(hlo, expected_pattern, match_times=1):
  matches = re.findall(expected_pattern, hlo)
  assert len(list(matches)) == match_times, hlo


def init_torch_distributed():
  succeeded = False
  retries = 5  # retry for port collision.
  dist.default_pg_timeout = timedelta(minutes=1)
  for _ in range(retries):
    try:
      filename = random_filename()
      port = random.randrange(49152, 65535)
      os.environ['XRT_WORKERS'] = f'localservice:0;grpc://localhost:{port}'
      # `world_size` needs to be 1, so that the rendezvous would only
      # wait for 1 participant, which is this program itself.
      # `init_method` is needed to provide a internal storage to comfort
      # torch.distributed.
      dist.init_process_group(
          'xla', rank=0, world_size=1, init_method='file:///tmp/' + filename)
      succeeded = True
      break
    except RuntimeError as err:
      os.remove('/tmp/' + filename)
      if 'Could not start gRPC server' in str(err):
        continue
      raise
  assert succeeded, f'Failed to init torch.distributed after {retries} retries'
  print('init_torch_distributed done')


@pytest.fixture(scope='module', autouse=True)
def setup(request):
  init_torch_distributed()


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


def test_xla_backend_exists():
  # torch_xla.distributed._register_xla_backend() should have been
  # automatically called.
  pg_xla_creator = dist.Backend.XLA
  assert pg_xla_creator is not None


def test_process_group_creation():
  pg_xla = get_process_group_xla(rank=1, size=2)
  assert pg_xla is not None
  assert pg_xla.rank() == 1
  assert pg_xla.size() == 2


def test_process_group_interface():
  # Verifies ProcessGroupXla has all ProcessGroup public APIs.
  # This test is needed because ProcessGroupXla cannot inherit from
  # ProcessGroup (see ProcessGroupXla doc), thus we have to use a unit
  # test to guarentee that ProcessGroupXla has all the required APIs.

  def is_public(n):
    return not n.startswith('_')

  process_group_apis = set(filter(is_public, dir(dist.ProcessGroup)))
  pg_xla = get_process_group_xla(rank=1, size=2)
  process_group_xla_apis = set(filter(is_public, dir(pg_xla)))
  missing = process_group_apis - process_group_xla_apis
  assert not missing, (f'ProcessGroupXla is missing the following '
                       'ProcessGroup APIs: {missing}')


def test_allreduce():
  device = xm.xla_device()
  tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
  pg_xla = get_process_group_xla(rank=511, size=1024)
  opts = dist.AllreduceOptions()
  opts.reduceOp = dist.ReduceOp.SUM
  all_reduce_pattern = r'%all\-reduce\.\d+ = .+ all\-reduce\('
  with xm_cc_op_intercepted('all_reduce'):
    work = pg_xla.allreduce([tensor], opts)
  hlo = torch_xla._XLAC._get_xla_tensors_hlo([tensor])
  hlo_matches(hlo, all_reduce_pattern)
  work.wait()


def test_allreduce_with_mesh():
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
    work = new_pg.allreduce([tensor], opts)
  hlo = torch_xla._XLAC._get_xla_tensors_hlo([tensor])
  hlo_matches(hlo, all_reduce_pattern)
  work.wait()


def test_allgather():
  device = xm.xla_device()
  tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
  pg_xla = get_process_group_xla(rank=3, size=8)
  output_tensors = [torch.zeros_like(tensor)] * 8
  all_gather_pattern = r'%all\-gather\.\d+ = .+ all\-gather\('
  with hlo_dump_monitored(all_gather_pattern, match_times=1):
    work = pg_xla.allgather([output_tensors], [tensor])
    work.wait()


def test_broadcast():
  device = xm.xla_device()
  tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
  pg_xla = get_process_group_xla(rank=0, size=8)
  opts = dist.BroadcastOptions()
  opts.rootRank = 0
  opts.rootTensor = 0
  # xla doesn't have broadcast. We use all_reduce to implement broadcast.
  all_reduce_pattern = r'%all\-reduce\.\d+ = .+ all\-reduce\('
  with xm_cc_op_intercepted('all_reduce'):
    work = pg_xla.broadcast([tensor], opts)
  hlo = torch_xla._XLAC._get_xla_tensors_hlo([tensor])
  hlo_matches(hlo, all_reduce_pattern)
  work.wait()


# Needed for ZeRO stage 1
def test_reduce_scatter():
  device = xm.xla_device()
  tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
  input_list = [tensor]
  output = torch.zeros_like(tensor)
  pg_xla = get_process_group_xla(rank=0, size=len(input_list))
  opts = dist.ReduceScatterOptions()
  opts.reduceOp = dist.ReduceOp.SUM
  # in tf2.6 reduce-scatter is called all-reduce-scatter.
  all_reduce_scatter_pattern = r'%reduce\-scatter\.\d+ = .+ reduce\-scatter\('
  with hlo_dump_monitored(all_reduce_scatter_pattern):
    work = pg_xla.reduce_scatter([output], [input_list], opts)
    work.wait()


def test_new_group_no_ranks():
  set_world_size(12)
  with new_group_barrier_disabled():
    pg = dist.new_group()
  assert isinstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
  assert pg.size() == get_world_size()


def test_new_group_horizontal():
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


def test_new_group_vertical():
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


def test_new_group_one_paticipant():
  set_world_size(12)

  ranks = [5]
  world_rank = 5
  set_world_rank(world_rank)
  with new_group_barrier_disabled():
    pg = dist.new_group(ranks=ranks)
  assert isinstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla), str(
      type(pg))
  assert pg.size() == 1
  assert pg.rank() == 0
  assert pg._mesh == [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
                      [11]]


def test_new_group_entire_world():
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


def test_new_group_invalid_horizontal():
  set_world_size(12)

  ranks = [4, 5, 6]
  world_rank = 5
  set_world_rank(world_rank)
  with new_group_barrier_disabled():
    with pytest.raises(ValueError):
      pg = dist.new_group(ranks=ranks)

  ranks = [0, 1, 2, 3, 4]
  world_rank = 2
  set_world_rank(world_rank)
  with new_group_barrier_disabled():
    with pytest.raises(ValueError):
      pg = dist.new_group(ranks=ranks)

  ranks = [7, 8, 9, 10]
  world_rank = 9
  set_world_rank(world_rank)
  with new_group_barrier_disabled():
    with pytest.raises(ValueError):
      pg = dist.new_group(ranks=ranks)


def test_new_group_invalid_vertical():
  set_world_size(12)

  ranks = [1, 5]
  world_rank = 5
  set_world_rank(world_rank)
  with new_group_barrier_disabled():
    with pytest.raises(ValueError):
      pg = dist.new_group(ranks=ranks)

  ranks = [4, 7, 10]
  world_rank = 4
  set_world_rank(world_rank)
  with new_group_barrier_disabled():
    with pytest.raises(ValueError):
      pg = dist.new_group(ranks=ranks)


def test_new_group_invalid_ranks():
  set_world_size(12)

  # unevenly distributed
  ranks = [1, 5, 10]
  world_rank = 5
  set_world_rank(world_rank)
  with new_group_barrier_disabled():
    with pytest.raises(ValueError):
      pg = dist.new_group(ranks=ranks)


def test_barrier():
  # nothing to verify. Just run it through.
  dist.barrier()


def test_unimplemented_ops():
  unimplemented_ops = (
      'reduce',
      'allgather_coalesced',
      'allreduce_coalesced',
      'alltoall',
      'alltoall_base',
      'gather',
      'scatter',
      'send',
      'recv',
      'recv_anysource',
      'monitored_barrier',
  )
  device = xm.xla_device()
  tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
  pg_xla = get_process_group_xla(rank=0, size=8)
  for op in unimplemented_ops:
    with pytest.raises(NotImplementedError):
      getattr(pg_xla, op)(tensor)


if __name__ == '__main__':
  pytest.main([__file__])
