import contextlib
import functools
import os
import re
from unittest import mock, skipIf

from absl.testing import absltest, parameterized
import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
from torch_xla import runtime as xr

from datetime import timedelta


def get_process_group_xla(rank, size):
  pg_xla_creator = dist.Backend._plugins['XLA'].creator_fn
  pg_xla = pg_xla_creator(
      prefix_store=None, rank=rank, size=size, timeout=timedelta(minutes=1))
  return pg_xla


def hlo_matches(hlo, expected_pattern, match_times=1):
  matches = re.findall(expected_pattern, hlo)
  assert len(list(matches)) == match_times, hlo


@contextlib.contextmanager
def new_group_barrier_disabled():
  with mock.patch.object(torch.distributed.distributed_c10d,
                         '_store_based_barrier'):
    yield


@contextlib.contextmanager
def patch_world(rank, size):
  assert isinstance(dist.group.WORLD,
                    torch_xla.distributed.xla_backend.ProcessGroupXla)
  with mock.patch.object(
      dist.group.WORLD, 'rank', return_value=rank), mock.patch.object(
          dist.group.WORLD, 'size', return_value=size):
    yield


@contextlib.contextmanager
def patch_world_with_xla_runtime(rank, size):
  assert isinstance(dist.group.WORLD,
                    torch_xla.distributed.xla_backend.ProcessGroupXla)
  with mock.patch.object(
      dist.group.WORLD, 'rank', return_value=rank), mock.patch.object(
          dist.group.WORLD, 'size', return_value=size), mock.patch.object(
              xr, 'global_ordinal', return_value=rank), mock.patch.object(
                  xr, 'world_size', return_value=size):
    yield


class XlaBackendTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    # Add no-op all-reduce ops to HLO
    os.environ['XLA_ALWAYS_ALLREDUCE'] = '1'
    dist.init_process_group('xla', init_method='xla://')

  def tearDown(self) -> None:
    # Purge all computations attached the device.
    torch_xla.sync()

  def test_xla_backend_exists(self):
    # torch_xla.distributed._register_xla_backend() should have been
    # automatically called.
    pg_xla_creator = dist.Backend.XLA
    self.assertIsNotNone(pg_xla_creator)

  def test_allreduce(self):
    device = torch_xla.device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    all_reduce_pattern = r'%all\-reduce\.\d+ = .+ all\-reduce\('
    dist.all_reduce(tensor)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([tensor])
    hlo_matches(hlo, all_reduce_pattern)

  @patch_world(rank=3, size=6)
  def test_allreduce_with_mesh(self):
    device = torch_xla.device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()

    pg_options = {'xla_pg_options': {'spmd': True}}
    ranks = [2, 3]
    with new_group_barrier_disabled():
      new_pg = dist.new_group(ranks=ranks, pg_options=pg_options)
    opts = dist.AllreduceOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    all_reduce_pattern = (r'%all\-reduce\.\d+ = .+ all\-reduce\(.+\), .*'
                          r'replica_groups=\{\{0,1\},\{2,3\},\{4,5\}\}')
    new_pg.allreduce([tensor], opts)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([tensor])
    hlo_matches(hlo, all_reduce_pattern)

  @patch_world(rank=3, size=8)
  def test_allgather(self):
    device = torch_xla.device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    output_tensors = [torch.zeros_like(tensor, device=device) for _ in range(8)]
    all_gather_pattern = r'%all\-gather\.\d+ = .+ all\-gather\('
    dist.all_gather(output_tensors, tensor)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo(output_tensors)
    hlo_matches(hlo, all_gather_pattern)

  @patch_world(rank=3, size=8)
  def test_all_scalar_allgather(self):
    device = torch_xla.device()
    tensor = torch.zeros((), device=device) + 1 + 2 * dist.get_rank()
    output_tensors = [torch.zeros_like(tensor, device=device) for _ in range(8)]
    all_gather_pattern = r'%all\-gather\.\d+ = .+ all\-gather\('
    dist.all_gather(output_tensors, tensor)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo(output_tensors)
    hlo_matches(hlo, all_gather_pattern)

  @patch_world(rank=3, size=8)
  def test_allgather_coalesced(self):
    device = torch_xla.device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    tensor2 = torch.arange(5, device=device) + 1 + 2 * dist.get_rank()
    pg_xla = get_process_group_xla(rank=3, size=8)
    output_tensors = [torch.zeros_like(tensor)] * 8
    output_tensors2 = [torch.zeros_like(tensor2)] * 8
    # because we set os.environ[xenv.WORLD_SIZE] = '1', here the outputs'
    # shapes will be same as the inputs' shapes.
    # Ex:  %all-gather.26 = (s64[2]{0}, s64[5]{0}) all-gather(s64[2]{0} %get-tuple-element.24, s64[5]{0} %get-tuple-element.25), replica_groups={}, dimensions={0}
    all_gather_pattern = (
        r'%all-gather\.\d+ = \(s64\[2]\{0}, s64\[5]\{0}\) '
        r'all-gather\(s64\[2]\{0} %.+\.\d+, s64\[5]\{0} %.+\.\d+\)')
    pg_xla.allgather_coalesced([output_tensors, output_tensors2],
                               [tensor, tensor2])
    hlo = torch_xla._XLAC._get_xla_tensors_hlo(output_tensors)
    hlo_matches(hlo, all_gather_pattern)

  def test_broadcast(self):
    device = torch_xla.device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    all_reduce_pattern = r'%all\-reduce\.\d+ = .+ all\-reduce\('
    dist.broadcast(tensor, 0)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([tensor])
    hlo_matches(hlo, all_reduce_pattern)

  # Needed for ZeRO stage 1
  def test_reduce_scatter(self):
    device = torch_xla.device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    input_list = [tensor]
    output = torch.zeros_like(tensor)
    reduce_scatter_pattern = r'%reduce\-scatter\.\d+ = .+ reduce\-scatter\('
    dist.reduce_scatter(output, input_list)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([output])
    hlo_matches(hlo, reduce_scatter_pattern)

  @skipIf(xr.device_type() == 'CPU',
          "UNIMPLEMENTED: ReduceScatter is not implemented on CPU.")
  def test_reduce_scatter_coalesced(self):
    device = torch_xla.device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    tensor2 = torch.arange(5, device=device) + 1 + 2 * dist.get_rank()
    input_tensors_list = [[tensor, tensor], [tensor2, tensor2]]
    output_list = [torch.zeros_like(tensor), torch.zeros_like(tensor2)]
    pg_xla = get_process_group_xla(rank=0, size=len(input_tensors_list[0]))
    opts = dist.ReduceScatterOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    reduce_scatter_pattern = (
        r'%reduce\-scatter\.\d+ = \(s64\[2]\{0}, s64\[5]\{0}, s64\[]\) '
        r'reduce\-scatter\(s64\[4]\{0} %.+\.\d+, s64\[10]\{0} %.+\.\d+, '
        r's64\[] %.+\.\d+\)')
    pg_xla.reduce_scatter_coalesced(output_list, input_tensors_list, opts)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo(output_list)
    hlo_matches(hlo, reduce_scatter_pattern)
    # purge all computations attached the device.
    torch_xla.sync()

  @patch_world(rank=0, size=12)
  def test_new_group_no_ranks(self):
    with new_group_barrier_disabled():
      pg = dist.new_group()
    self.assertIsInstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
    self.assertEqual(pg.size(), dist.get_world_size())

  def test_new_group_horizontal(self):
    pg_options = {'xla_pg_options': {'spmd': True}}
    with patch_world(rank=5, size=12):
      ranks = [4, 5, 6, 7]
      with new_group_barrier_disabled():
        pg = dist.new_group(ranks=ranks, pg_options=pg_options)
      self.assertIsInstance(pg,
                            torch_xla.distributed.xla_backend.ProcessGroupXla)
      self.assertEqual(pg.size(), len(ranks))
      self.assertEqual(pg.rank(), ranks.index(5))
      self.assertListEqual(pg._mesh,
                           [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])

    with patch_world(rank=2, size=12):
      ranks = [0, 1, 2, 3]
      with new_group_barrier_disabled():
        pg = dist.new_group(ranks=ranks, pg_options=pg_options)
      self.assertIsInstance(pg,
                            torch_xla.distributed.xla_backend.ProcessGroupXla)
      self.assertEqual(pg.size(), len(ranks))
      self.assertEqual(pg.rank(), ranks.index(2))
      self.assertListEqual(pg._mesh,
                           [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])

    with patch_world(rank=11, size=12):
      ranks = [8, 9, 10, 11]
      with new_group_barrier_disabled():
        pg = dist.new_group(ranks=ranks, pg_options=pg_options)
      self.assertIsInstance(pg,
                            torch_xla.distributed.xla_backend.ProcessGroupXla)
      self.assertEqual(pg.size(), len(ranks))
      self.assertEqual(pg.rank(), ranks.index(11))
      self.assertListEqual(pg._mesh,
                           [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])

  def test_new_group_vertical(self):
    pg_options = {'xla_pg_options': {'spmd': True}}
    with patch_world(rank=5, size=12):
      ranks = [1, 5, 9]
      with new_group_barrier_disabled():
        pg = dist.new_group(ranks=ranks, pg_options=pg_options)
      self.assertIsInstance(pg,
                            torch_xla.distributed.xla_backend.ProcessGroupXla)
      self.assertEqual(pg.size(), len(ranks))
      self.assertEqual(pg.rank(), ranks.index(5))
      self.assertListEqual(pg._mesh,
                           [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]])

    with patch_world(rank=0, size=12):
      ranks = [0, 4, 8]
      with new_group_barrier_disabled():
        pg = dist.new_group(ranks=ranks, pg_options=pg_options)
      self.assertIsInstance(pg,
                            torch_xla.distributed.xla_backend.ProcessGroupXla)
      self.assertEqual(pg.size(), len(ranks))
      self.assertEqual(pg.rank(), ranks.index(0))
      self.assertListEqual(pg._mesh,
                           [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]])

    with patch_world(rank=11, size=12):
      ranks = [3, 7, 11]
      with new_group_barrier_disabled():
        pg = dist.new_group(ranks=ranks, pg_options=pg_options)
      self.assertIsInstance(pg,
                            torch_xla.distributed.xla_backend.ProcessGroupXla)
      self.assertEqual(pg.size(), len(ranks))
      self.assertEqual(pg.rank(), ranks.index(11))
      self.assertListEqual(pg._mesh,
                           [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]])

  @patch_world(rank=5, size=12)
  def test_new_group_one_paticipant(self):

    pg_options = {'xla_pg_options': {'spmd': True}}
    ranks = [5]
    with new_group_barrier_disabled():
      pg = dist.new_group(ranks=ranks, pg_options=pg_options)
    self.assertIsInstance(pg, torch_xla.distributed.xla_backend.ProcessGroupXla)
    self.assertEqual(pg.size(), 1)
    self.assertEqual(pg.rank(), 0)
    self.assertListEqual(
        pg._mesh,
        [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]])

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
    pg_options = {'xla_pg_options': {'spmd': True}}
    with patch_world(rank=5, size=12):
      ranks = [4, 5, 6]
      with new_group_barrier_disabled():
        with self.assertRaises(ValueError):
          dist.new_group(ranks=ranks, pg_options=pg_options)

    with patch_world(rank=2, size=12):
      ranks = [0, 1, 2, 3, 4]
      with new_group_barrier_disabled():
        with self.assertRaises(ValueError):
          dist.new_group(ranks=ranks, pg_options=pg_options)

    with patch_world(rank=9, size=12):
      ranks = [7, 8, 9, 10]
      with new_group_barrier_disabled():
        with self.assertRaises(ValueError):
          dist.new_group(ranks=ranks, pg_options=pg_options)

  def test_new_group_invalid_vertical(self):
    pg_options = {'xla_pg_options': {'spmd': True}}
    with patch_world(rank=5, size=12):
      ranks = [1, 5]
      with new_group_barrier_disabled():
        with self.assertRaises(ValueError):
          dist.new_group(ranks=ranks, pg_options=pg_options)

    with patch_world(rank=4, size=12):
      ranks = [4, 7, 10]
      with new_group_barrier_disabled():
        with self.assertRaises(ValueError):
          dist.new_group(ranks=ranks, pg_options=pg_options)

  def test_new_group_invalid_ranks(self):
    # unevenly distributed
    pg_options = {'xla_pg_options': {'spmd': True}}
    with patch_world(rank=5, size=12):
      ranks = [1, 5, 10]
      with new_group_barrier_disabled():
        with self.assertRaises(ValueError):
          dist.new_group(ranks=ranks, pg_options=pg_options)

  def test_barrier(self):
    # nothing to verify. Just run it through.
    dist.barrier()

  @parameterized.parameters(
      'allreduce_coalesced',
      'recv_anysource',
      'monitored_barrier',
  )
  def test_unimplemented_op(self, op):
    device = torch_xla.device()
    tensor = torch.arange(2, device=device) + 1 + 2 * dist.get_rank()
    pg_xla = dist.group.WORLD
    self.assertIsInstance(pg_xla,
                          torch_xla.distributed.xla_backend.ProcessGroupXla)
    with self.assertRaises(NotImplementedError):
      getattr(pg_xla, op)(tensor)

  @patch_world_with_xla_runtime(rank=0, size=2)
  def test_broadcast_single_rank_group_rank0(self):
    """Test broadcast in single-member process group for rank 0"""
    device = torch_xla.device()

    with new_group_barrier_disabled():
      tp = dist.new_group(ranks=[0])

    # Create flags tensor with initial values (simulating rank 0's values)
    flags = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=device)

    # Broadcast within the single-member group (should be a no-op but shouldn't crash)
    dist.broadcast(flags, src=0, group=tp)

    # Values should remain unchanged since it's a single-member group
    self.assertAlmostEqual(flags[0].item(), 0.1, places=5)
    self.assertAlmostEqual(flags[1].item(), 0.2, places=5)
    self.assertAlmostEqual(flags[2].item(), 0.3, places=5)

    # Verify the process group properties
    self.assertEqual(dist.get_rank(group=tp), 0)
    self.assertEqual(dist.get_world_size(group=tp), 1)

  @patch_world_with_xla_runtime(rank=1, size=2)
  def test_broadcast_single_rank_group_rank1(self):
    """Test broadcast in single-member process group for rank 1"""
    device = torch_xla.device()

    with new_group_barrier_disabled():
      tp = dist.new_group(ranks=[1])

    # Create flags tensor with initial values (simulating rank 1's values)
    flags = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=device)

    # Broadcast within the single-member group (should be a no-op but shouldn't crash)
    dist.broadcast(flags, src=1, group=tp)

    # Values should remain unchanged since it's a single-member group
    self.assertAlmostEqual(flags[0].item(), 0.1, places=5)
    self.assertAlmostEqual(flags[1].item(), 0.2, places=5)
    self.assertAlmostEqual(flags[2].item(), 0.3, places=5)

    # Verify the process group properties
    self.assertEqual(dist.get_rank(group=tp),
                     0)  # Local rank in single-member group is 0
    self.assertEqual(dist.get_world_size(group=tp), 1)

  @patch_world_with_xla_runtime(rank=0, size=2)
  def test_broadcast_global_rank_conversion_single_member(self):
    """Test that global rank conversion works correctly for single-member groups"""
    device = torch_xla.device()

    # Create single-member group for rank 0
    with new_group_barrier_disabled():
      tp = dist.new_group(ranks=[0])

    flags = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=device)

    # Get the ProcessGroupXla instance to test directly
    self.assertIsInstance(tp, torch_xla.distributed.xla_backend.ProcessGroupXla)

    # Test broadcast options - local rank 0 should map to global rank 0
    opts = dist.BroadcastOptions()
    opts.rootRank = 0
    opts.rootTensor = 0

    # This should work without variable name errors
    work = tp.broadcast([flags], opts)
    self.assertIsNotNone(work)

    # Values should be preserved
    self.assertAlmostEqual(flags[0].item(), 0.1, places=5)
    self.assertAlmostEqual(flags[1].item(), 0.2, places=5)
    self.assertAlmostEqual(flags[2].item(), 0.3, places=5)


if __name__ == '__main__':
  if xr.device_type() != 'CPU':
    print(f"Skipping XLA backend unit test as this test doesn't exercise"
          "{xr.pjrt_device}-specific behaviors.")
    exit(0)

  absltest.main()
