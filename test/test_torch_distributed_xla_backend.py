import contextlib
import functools
import re
from unittest import mock

from absl.testing import absltest, parameterized
import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
from torch_xla.experimental import pjrt

def hlo_matches(hlo, expected_pattern, match_times=1):
  matches = re.findall(expected_pattern, hlo)
  assert len(list(matches)) == match_times, hlo


@contextlib.contextmanager
def new_group_barrier_disabled():
  with mock.patch.object(torch.distributed.distributed_c10d, '_store_based_barrier'):
    yield

@contextlib.contextmanager
def always_intercore_reduce():
  OriginalCollectiveContext = xm.CollectiveContext

  @functools.wraps(OriginalCollectiveContext)
  def MockCollectiveContext(groups=None):
    ctx = OriginalCollectiveContext()
    ctx.requires_intercore_reduce = True
    ctx.intercore_group = groups
    return ctx

  with mock.patch.object(xm, 'CollectiveContext', new=MockCollectiveContext):
    yield

@contextlib.contextmanager
def patch_world(rank, size):
  assert isinstance(dist.group.WORLD, torch_xla.distributed.xla_backend.ProcessGroupXla)
  with mock.patch.object(dist.group.WORLD, 'rank', return_value=rank), mock.patch.object(dist.group.WORLD, 'size', return_value=size):
    yield


class XlaBackendTest(parameterized.TestCase):

  def tearDown(self) -> None:
    # Purge all computations attached the device.
    xm.mark_step()

  def test_xla_backend_exists(self):
    # torch_xla.distributed._register_xla_backend() should have been
    # automatically called.
    pg_xla_creator = dist.Backend.XLA
    self.assertIsNotNone(pg_xla_creator)

  @always_intercore_reduce()
  def test_allreduce(self):
    device = xm.xla_device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    all_reduce_pattern = r'%all\-reduce\.\d+ = .+ all\-reduce\('
    dist.all_reduce(tensor)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([tensor])
    hlo_matches(hlo, all_reduce_pattern)

  @always_intercore_reduce()
  @patch_world(rank=3, size=6)
  def test_allreduce_with_mesh(self):
    device = xm.xla_device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()

    ranks = [2, 3]
    with new_group_barrier_disabled():
      new_pg = dist.new_group(ranks=ranks)
    opts = dist.AllreduceOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    all_reduce_pattern = (r'%all\-reduce\.\d+ = .+ all\-reduce\(.+\), .*'
                          r'replica_groups=\{\{0,1\},\{2,3\},\{4,5\}\}')
    new_pg.allreduce([tensor], opts)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([tensor])
    hlo_matches(hlo, all_reduce_pattern)

  @always_intercore_reduce()
  @patch_world(rank=3, size=8)
  def test_allgather(self):
    device = xm.xla_device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    output_tensors = [torch.zeros_like(tensor, device=device) for _ in range(8)]
    all_gather_pattern = r'%all\-gather\.\d+ = .+ all\-gather\('
    dist.all_gather(output_tensors, tensor)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo(output_tensors)
    hlo_matches(hlo, all_gather_pattern)

  @always_intercore_reduce()
  def test_broadcast(self):
    device = xm.xla_device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    all_reduce_pattern = r'%all\-reduce\.\d+ = .+ all\-reduce\('
    dist.broadcast(tensor, 0)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([tensor])
    hlo_matches(hlo, all_reduce_pattern)

  # Needed for ZeRO stage 1
  @always_intercore_reduce()
  def test_reduce_scatter(self):
    device = xm.xla_device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    input_list = [tensor]
    output = torch.zeros_like(tensor)
    reduce_scatter_pattern = r'%reduce\-scatter\.\d+ = .+ reduce\-scatter\('
    dist.reduce_scatter(output, input_list)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([output])
    hlo_matches(hlo, reduce_scatter_pattern)

  # def test_send(self):
  #   device = xm.xla_device()
  #   tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
  #   input_list = [tensor]
  #   set_world_size(6)
  #   ranks = [0, 3]
  #   world_rank = 0
  #   set_world_rank(world_rank)

  #   torch_xla.distributed.xla_backend.ProcessGroupXla.make_send_channel_id = (
  #       lambda self, dst_rank, tag: dst_rank * 2)

  #   with new_group_barrier_disabled():
  #     pg_xla = dist.new_group(ranks=ranks)

  #   send_pattern = r'%send\.\d+ = .+ send\(.+\), channel_id=2'
  #   senddone_pattern = r'%send\-done\.\d+ = .+ send\-done\(.+\), channel_id=2'
  #   # seeing 'Send is not implemented on CPU' means we have successfully
  #   # generated `send` in the HLO.
  #   with self.assertRaises(RuntimeError) as cm:
  #     pg_xla.send(input_list, 1)
  #     hlo = torch_xla._XLAC._get_xla_tensors_hlo(input_list)
  #     hlo_matches(hlo, send_pattern)
  #     hlo_matches(hlo, senddone_pattern)
  #     xm.mark_step()
  #   assert 'UNIMPLEMENTED: Send is not implemented on CPU.' in str(
  #       cm.exception), str(cm.exception)
  #   # reset token to clean up the mess after the RuntimeError.
  #   xm.set_replication(device, [])

#   def test_recv(self):
#     device = xm.xla_device()
#     tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
#     output_list = [tensor]
#     set_world_size(6)
#     ranks = [0, 3]
#     world_rank = 0
#     set_world_rank(world_rank)

#     torch_xla.distributed.xla_backend.ProcessGroupXla.make_recv_channel_id = (
#         lambda self, src_rank, tag: src_rank * 3)

#     with new_group_barrier_disabled():
#       pg_xla = dist.new_group(ranks=ranks)

#     recv_pattern = r'%recv\.\d+ = .+ recv\(.+\), channel_id=3'
#     recvdone_pattern = r'%recv\-done\.\d+ = .+ recv\-done\(.+\), channel_id=3'
#     # seeing 'recv is not implemented on CPU' means we have successfully
#     # generated `recv` in the HLO.
#     with self.assertRaises(RuntimeError) as cm:
#       pg_xla.recv(output_list, 1)
#       hlo = torch_xla._XLAC._get_xla_tensors_hlo(output_list)
#       hlo_matches(hlo, recv_pattern)
#       hlo_matches(hlo, recvdone_pattern)
#       xm.mark_step()
#     assert 'UNIMPLEMENTED: Recv is not implemented on CPU.' in str(
#         cm.exception), str(cm.exception)
#     # reset token to clean up the mess after the RuntimeError.
#     xm.set_replication(device, [])

  @patch_world(rank=0, size=12)
  def test_new_group_no_ranks(self):
    with new_group_barrier_disabled():
      pg = dist.new_group()
    self.assertIsInstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
    self.assertEqual(pg.size(), dist.get_world_size())

  def test_new_group_horizontal(self):
    with patch_world(rank=5, size=12):
      ranks = [4, 5, 6, 7]
      with new_group_barrier_disabled():
        pg = dist.new_group(ranks=ranks)
      self.assertIsInstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
      self.assertEqual(pg.size(), len(ranks))
      self.assertEqual(pg.rank(), ranks.index(5))
      self.assertListEqual(pg._mesh, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])

    with patch_world(rank=2, size=12):
      ranks = [0, 1, 2, 3]
      with new_group_barrier_disabled():
        pg = dist.new_group(ranks=ranks)
      self.assertIsInstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
      self.assertEqual(pg.size(), len(ranks))
      self.assertEqual(pg.rank(),  ranks.index(2))
      self.assertListEqual(pg._mesh, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])

    with patch_world(rank=11, size=12):
      ranks = [8, 9, 10, 11]
      with new_group_barrier_disabled():
        pg = dist.new_group(ranks=ranks)
      self.assertIsInstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
      self.assertEqual(pg.size(), len(ranks))
      self.assertEqual(pg.rank(), ranks.index(11))
      self.assertListEqual(pg._mesh, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])

  def test_new_group_vertical(self):
    with patch_world(rank=5, size=12):
      ranks = [1, 5, 9]
      with new_group_barrier_disabled():
        pg = dist.new_group(ranks=ranks)
      self.assertIsInstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
      self.assertEqual(pg.size(), len(ranks))
      self.assertEqual(pg.rank(), ranks.index(5))
      self.assertListEqual(pg._mesh, [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]])

    with patch_world(rank=0, size=12):
      ranks = [0, 4, 8]
      with new_group_barrier_disabled():
        pg = dist.new_group(ranks=ranks)
      self.assertIsInstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
      self.assertEqual(pg.size(), len(ranks))
      self.assertEqual(pg.rank(), ranks.index(0))
      self.assertListEqual(pg._mesh, [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]])

    with patch_world(rank=11, size=12):
      ranks = [3, 7, 11]
      with new_group_barrier_disabled():
        pg = dist.new_group(ranks=ranks)
      self.assertIsInstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
      self.assertEqual(pg.size(), len(ranks))
      self.assertEqual(pg.rank(), ranks.index(11))
      self.assertListEqual(pg._mesh, [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]])

  @patch_world(rank=5, size=12)
  def test_new_group_one_paticipant(self):

    ranks = [5]
    with new_group_barrier_disabled():
      pg = dist.new_group(ranks=ranks)
    self.assertIsInstance(pg,
                      torch_xla.distributed.xla_backend.ProcessGroupXla)
    self.assertEqual(pg.size(), 1)
    self.assertEqual(pg.rank(), 0)
    self.assertListEqual(pg._mesh, [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
                        [11]])

  @patch_world(rank=5, size=12)
  def test_new_group_entire_world(self):
    ranks = range(12)
    with new_group_barrier_disabled():
      pg = dist.new_group(ranks=ranks)
    self.assertIsInstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
    self.assertEqual(pg.size(), 12)
    self.assertEqual(pg.rank(), 5)
    self.assertListEqual(pg._mesh, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])

  def test_new_group_invalid_horizontal(self):
    with patch_world(rank=5, size=12):
      ranks = [4, 5, 6]
      with new_group_barrier_disabled():
        with self.assertRaises(ValueError):
          dist.new_group(ranks=ranks)

    with patch_world(rank=2, size=12):
      ranks = [0, 1, 2, 3, 4]
      with new_group_barrier_disabled():
        with self.assertRaises(ValueError):
          dist.new_group(ranks=ranks)

    with patch_world(rank=9, size=12):
      ranks = [7, 8, 9, 10]
      with new_group_barrier_disabled():
        with self.assertRaises(ValueError):
          dist.new_group(ranks=ranks)

  def test_new_group_invalid_vertical(self):
    with patch_world(rank=5, size=12):
      ranks = [1, 5]
      with new_group_barrier_disabled():
        with self.assertRaises(ValueError):
          dist.new_group(ranks=ranks)

    with patch_world(rank=4, size=12):
      ranks = [4, 7, 10]
      with new_group_barrier_disabled():
        with self.assertRaises(ValueError):
          dist.new_group(ranks=ranks)

  def test_new_group_invalid_ranks(self):
    # unevenly distributed
    with patch_world(rank=5, size=12):
      ranks = [1, 5, 10]
      with new_group_barrier_disabled():
        with self.assertRaises(ValueError):
          dist.new_group(ranks=ranks)

  def test_barrier(self):
    # nothing to verify. Just run it through.
    dist.barrier()

  @parameterized.parameters(
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
  def test_unimplemented_op(self, op):
    device = xm.xla_device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    pg_xla = dist.group.WORLD
    self.assertIsInstance(pg_xla, torch_xla.distributed.xla_backend.ProcessGroupXla)
    with self.assertRaises(NotImplementedError):
      getattr(pg_xla, op)(tensor)


if __name__ == '__main__':
  if pjrt.device_type() != 'CPU':
    print(f"Skipping XLA backend unit test as this test doesn't exercise"
           "{pjrt.pjrt_device}-specific behaviors.")
    exit(0)

  dist.init_process_group('xla', rank=0, world_size=1, init_method='tcp://localhost:6789')
  absltest.main()
