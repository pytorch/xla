import functools
import os
import signal
import sys
import tempfile
import test_xla_sharding_base
import threading
import time
import unittest

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs

from torch.distributed.checkpoint._fsspec_filesystem import *
from collections.abc import Iterable

from torch.distributed.checkpoint.default_planner import (
    create_default_local_save_plan,
    create_default_global_save_plan,
)
from torch_xla.experimental.distributed_checkpoint import SPMDLoadPlanner, SPMDSavePlanner, CheckpointManager, prime_optimizer
from torch_xla.experimental.distributed_checkpoint._helpers import (
    _sharded_cpu_state_dict, _CpuShards, _is_sharded_tensor)


# Wrapper to manage a temporary directory for the wrapped test
def run_with_tmpdir(f):

  @functools.wraps(f)
  def run(*args, **kwargs):
    with tempfile.TemporaryDirectory() as tmpdir:
      kwargs.setdefault('tmpdir', tmpdir)
      f(*args, **kwargs)

  return run


class DistributedCheckpointTestBase(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  def _get_sharded_model(self, mesh_shape=None):
    # Return a sharded SimpleLinear model with fc1.weight sharded and
    # fc2.weight explicitly replicated
    mesh_shape = mesh_shape or (1, self.n_devices)
    model = self.SimpleLinear().to(xm.xla_device())
    mesh = self._get_mesh(mesh_shape)
    xs.mark_sharding(model.fc1.weight, mesh, (0, 1))
    xs.mark_sharding(model.fc2.weight, mesh, (None, None))
    return model

  def _get_default_local_metadata(self):
    # Create a default Metadata instance for a SimpleLinear model by using
    # the default local save plan on the model's CPU state_dict
    cpu_state_dict = self.SimpleLinear().state_dict()
    local_plan = create_default_local_save_plan(cpu_state_dict, True)
    _, md = create_default_global_save_plan([local_plan])
    return md

  def _same_shard_data(self, shards, others) -> bool:
    for a, b in zip(shards, others):
      if not torch.allclose(a.data, b.data):
        return False
    return True

  def _assert_same_state_dict(self, sd1, sd2, keypath=""):
    assert type(sd1) == type(
        sd2), f"Different types in state_dict: {sd1} vs {sd2}"

    if isinstance(sd1, torch.Tensor):
      assert sd1.device == sd2.device, f"Tensors on different devices at {keypath}: {sd1} vs {sd2}"
      if sd1.device == xm.xla_device():
        sharding1 = torch_xla._XLAC._get_xla_sharding_spec(sd1)
        sharding2 = torch_xla._XLAC._get_xla_sharding_spec(sd2)
        assert sharding1 == sharding2, f"Different sharding on tensors at {keypath}: {sharding1} vs {sharding2}"
      assert torch.equal(
          sd1, sd2), f"Different tensors at {keypath}:\n{sd1} vs {sd2}"

    elif isinstance(sd1, dict):
      assert sd1.keys() == sd2.keys(
      ), f"Different keys at {keypath}: {sd1} vs {sd2}"
      for key in sd1:
        self._assert_same_state_dict(
            sd1[key], sd2[key], keypath=f'{keypath}.{key}')

    elif isinstance(sd1, Iterable):
      for ind, (a, b) in enumerate(zip(sd1, sd2)):
        self._assert_same_state_dict(a, b, keypath=f'{keypath}[{ind}]')

    else:
      assert sd1 == sd2, f"Different value at {keypath}: {sd1} vs {sd2}"


class EndToEndCheckpointTest(DistributedCheckpointTestBase):

  def _save_and_restore(self,
                        model_in,
                        model_out,
                        save_planner=None,
                        load_planner=None,
                        storage_writer_cls=dist_cp.FileSystemWriter,
                        storage_reader_cls=dist_cp.FileSystemReader,
                        is_sharded_cpu_state_dict=False,
                        chkpt_path=None):
    """
    Checkpoint model_in using the provided save_planner and load into model_out
    using the provided load_planner, and assert model_out equals model_in after
    the load. If either planner is not specified, the DefaultPlanner is used.
    """
    chkpt_path = chkpt_path or tempfile.mkdtemp()

    # Save an unsharded model using the provided save planner
    model_in_state_dict = model_in.state_dict()
    if is_sharded_cpu_state_dict:
      model_in_state_dict = _sharded_cpu_state_dict(model_in_state_dict)
    model_out_state_dict = model_out.state_dict()
    dist_cp.save(
        state_dict=model_in_state_dict,
        storage_writer=storage_writer_cls(
            chkpt_path,
            sync_files=False,
            per_thread_copy_ahead=0,
        ),
        planner=save_planner,
    )
    # Load the checkpoint using the provided load planner
    for p1, p2 in zip(model_in.parameters(), model_out.parameters()):
      self.assertFalse(torch.allclose(p1.cpu(), p2.cpu()))

    dist_cp.load(
        state_dict=model_out_state_dict,
        storage_reader=storage_reader_cls(chkpt_path),
        planner=load_planner,
    )
    for p1, p2 in zip(model_in.parameters(), model_out.parameters()):
      self.assertTrue(torch.allclose(p1.cpu(), p2.cpu()))

  def test_resharding_unsharded_to_sharded(self):
    # Save an unsharded model using the DefaultSavePlanner and load into a
    # sharded model using the SPMDLoadPlanner
    model = self.SimpleLinear().to(xm.xla_device())
    sharded_model = self._get_sharded_model()
    self._save_and_restore(model, sharded_model, load_planner=SPMDLoadPlanner())

  def test_resharding_sharded_to_unsharded(self):
    for chkpt_on_cpu in [True, False]:
      with self.subTest(chkpt_on_cpu):
        model = self.SimpleLinear().to(xm.xla_device())
        sharded_model = self._get_sharded_model()
        self._save_and_restore(
            sharded_model,
            model,
            save_planner=SPMDSavePlanner(),
            is_sharded_cpu_state_dict=chkpt_on_cpu)

  @unittest.skipIf(xr.global_runtime_device_count() == 1,
                   "Multiple devices needed to change mesh")
  def test_resharding_different_device_mesh(self):
    dim = self.n_devices // 2
    model1 = self._get_sharded_model(mesh_shape=(dim, self.n_devices // dim))
    model2 = self._get_sharded_model(mesh_shape=(self.n_devices, 1))
    self._save_and_restore(
        model1,
        model2,
        save_planner=SPMDSavePlanner(),
        load_planner=SPMDLoadPlanner())

  @unittest.skipIf(xr.global_runtime_device_count() == 1,
                   "Multiple devices needed to change mesh")
  def test_resharding_transpose_device_mesh(self):
    dim = self.n_devices // 2
    model1 = self._get_sharded_model(mesh_shape=(dim, self.n_devices // dim))
    model2 = self._get_sharded_model(mesh_shape=(self.n_devices // dim, dim))
    self._save_and_restore(
        model1,
        model2,
        save_planner=SPMDSavePlanner(),
        load_planner=SPMDLoadPlanner())

  @unittest.skipIf(xr.global_runtime_device_count() == 1,
                   "Multiple devices needed to change mesh")
  def test_padded_tensor(self):
    # Use a linear layer with shape not divisible by the number of devices.
    model1 = torch.nn.Linear(127, 63).to('xla')
    model2 = torch.nn.Linear(127, 63).to('xla')
    mesh = xs.Mesh(range(self.n_devices), (self.n_devices,))
    # Transpose the sharding to induce resharding in the restore path
    xs.mark_sharding(model1.weight, mesh, (0, None))
    xs.mark_sharding(model2.weight, mesh, (None, 0))
    self._save_and_restore(
        model1,
        model2,
        save_planner=SPMDSavePlanner(),
        load_planner=SPMDLoadPlanner())

  @unittest.skipUnless('CHKPT_PATH' in os.environ,
                       'CHKPT_PATH must be set for multihost checkpoint')
  def test_multihost_checkpoint(self):
    torch.manual_seed(42)

    # Initialize the default CPU process group.
    import torch_xla.distributed.xla_backend
    dist.init_process_group(backend='gloo', init_method='xla://')

    model1 = self._get_sharded_model(mesh_shape=(1, self.n_devices))
    model2 = self._get_sharded_model(mesh_shape=(self.n_devices, 1))
    # Take the checkpoint, writing to the path configured in the environment.
    self._save_and_restore(
        model1,
        model2,
        save_planner=SPMDSavePlanner(),
        load_planner=SPMDLoadPlanner(),
        storage_writer_cls=FsspecWriter,
        storage_reader_cls=FsspecReader,
        chkpt_path=os.environ['CHKPT_PATH'])

    # Destroy the CPU process group after the test
    dist.destroy_process_group()


class SPMDLoadPlannerTest(DistributedCheckpointTestBase):

  def _get_load_planner(self, model):
    # Create an SPMDLoadPlanner for the given model.
    md = self._get_default_local_metadata()
    planner = SPMDLoadPlanner()
    planner.set_up_planner(model.state_dict(), md, True)
    return planner

  def test_state_dict_separation(self):
    model = self._get_sharded_model()
    planner = self._get_load_planner(model)
    if self.n_devices > 1:
      # The state_dict should be flattened and separated
      self.assertCountEqual(planner.sharded_state_dict, ['fc1.weight'])
      # `fc2.weight` should be in the unsharded_state_dict despite having
      # an explicit mark_sharding call because it is replicated.
      self.assertCountEqual(planner.unsharded_state_dict,
                            ['fc1.bias', 'fc2.weight', 'fc2.bias'])
    else:
      # With a single device, no tensors are sharded.
      self.assertCountEqual(
          planner.unsharded_state_dict,
          ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])

  def test_local_load_plan(self):
    model = self._get_sharded_model()
    planner = self._get_load_planner(model)
    plan = planner.create_local_plan()
    parameter_count = len(list(model.parameters()))
    if self.n_devices > 1:
      # When the model is sharded across devices, fc1.weight will result in
      # self.n_devices ReadItems while all other tensors result in a single
      # ReadItem because the checkpoint metadata is unsharded.
      sharded_read_items = [
          ri for ri in plan.items if ri.dest_index.fqn == 'fc1.weight'
      ]
      self.assertEqual(self.n_devices, len(sharded_read_items))
      # Every other parameter should have a single ReadItem
      unsharded_read_items = set(plan.items) - set(sharded_read_items)
      self.assertEqual(parameter_count - 1, len(unsharded_read_items))
    else:
      # If unsharded, there should be a single ReadItem per model parameter
      self.assertEqual(parameter_count, len(plan.items))

  @unittest.skipIf(xr.global_runtime_device_count() == 1,
                   "Multiple devices required to shard tensors")
  def test_resolve_and_commit_sharded_tensor(self):
    model = self._get_sharded_model()
    planner = self._get_load_planner(model)
    plan = planner.create_local_plan()

    xtensor = xs.wrap_if_sharded(model.fc1.weight)
    old_shards = xtensor.local_shards
    sharded_read_items = [
        ri for ri in plan.items if ri.dest_index.fqn == 'fc1.weight'
    ]
    self.assertEqual(self.n_devices, len(sharded_read_items))
    for read_item in sharded_read_items:
      # Before all ReadItems have been processed, the tensor shards should not
      # be updated
      self.assertTrue(self._same_shard_data(xtensor.local_shards, old_shards))
      tensor = planner.resolve_tensor(read_item)
      tensor *= -1
      planner.commit_tensor(read_item, tensor)
    # After all ReadItems are processed, the local_shards should reflect the new
    # values
    self.assertFalse(self._same_shard_data(xtensor.local_shards, old_shards))


class SPMDSavePlannerTest(DistributedCheckpointTestBase):

  def _get_save_planner(self, model, is_sharded_cpu_state_dict=False):
    # Create an SPMDSavePlanner for the given model.
    planner = SPMDSavePlanner()
    if not is_sharded_cpu_state_dict:
      planner.set_up_planner(model.state_dict(), True)
    else:
      planner.set_up_planner(_sharded_cpu_state_dict(model.state_dict()), True)
    return planner

  def _planner_assertions(self, planner):
    if self.n_devices > 1:
      # The state_dict should be flattened and separated
      self.assertCountEqual(planner.sharded_state_dict, ['fc1.weight'])
      # `fc2.weight` should be in the unsharded_state_dict despite having
      # an explicit mark_sharding call because it is replicated.
      self.assertCountEqual(planner.unsharded_state_dict,
                            ['fc1.bias', 'fc2.weight', 'fc2.bias'])
    else:
      # With a single device, no tensors are sharded.
      self.assertCountEqual(
          planner.unsharded_state_dict,
          ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])

  def test_state_dict_separation(self):
    model = self._get_sharded_model()
    planner = self._get_save_planner(model)
    self._planner_assertions(planner)

  def test_save_state_dict_with_cpu_shards(self):
    model = self._get_sharded_model()
    planner = self._get_save_planner(model, is_sharded_cpu_state_dict=True)
    self._planner_assertions(planner)
    if self.n_devices > 1:
      self.assertTrue(
          isinstance(planner.sharded_state_dict['fc1.weight'], _CpuShards))

  @unittest.skipUnless(xr.global_runtime_device_count() > 1,
                       "Multiple devices required for sharded test")
  def test_cpu_state_dict_flattening(self):
    # In the case of a nested state_dict with fully sharded parameters,
    # _CpuShards should be treated as terminal nodes.
    t = torch.randn(128, 128).to(xm.xla_device())
    mesh = self._get_mesh((self.n_devices, 1))
    xs.mark_sharding(t, mesh, (0, 1))
    state_dict = _sharded_cpu_state_dict({'model': {'weight': t}})
    planner = SPMDSavePlanner()
    planner.set_up_planner(state_dict, True)
    # model.weight should be flattened and tracked in the sharded state dict.
    self.assertCountEqual(planner.sharded_state_dict, ["model.weight"])

  def test_local_save_plan(self):

    def _write_item_assertions(plan, n_devices, parameter_count):
      if n_devices > 1:
        # When the model is sharded across devices, fc1.weight will result in
        # self.n_devices WriteItems while all other tensors result in a single
        # WriteItem.
        sharded_write_items = [
            wi for wi in plan.items if wi.index.fqn == 'fc1.weight'
        ]
        self.assertEqual(self.n_devices, len(sharded_write_items))
        # Every other parameter should have a single WriteItem
        unsharded_write_items = [
            x for x in plan.items if x not in sharded_write_items
        ]
        self.assertEqual(parameter_count - 1, len(unsharded_write_items))
      else:
        self.assertEqual(parameter_count, len(plan.items))
      # If unsharded, there should be a single WriteItem per model parameter

    for chkpt_on_cpu in [True, False]:
      with self.subTest(chkpt_on_cpu):
        model = self._get_sharded_model()
        planner = self._get_save_planner(model, chkpt_on_cpu)
        plan = planner.create_local_plan()
        parameter_count = len(list(model.parameters()))
        _write_item_assertions(plan, self.n_devices, parameter_count)

  @unittest.skipIf(xr.global_runtime_device_count() == 1,
                   "Multiple devices required to shard tensors")
  def test_resolve_shard_data(self):
    model = self._get_sharded_model()
    planner = self._get_save_planner(model)
    plan = planner.create_local_plan()

    shards = xs.wrap_if_sharded(model.fc1.weight).local_shards
    sharded_write_items = [
        wi for wi in plan.items if wi.index.fqn == 'fc1.weight'
    ]
    for write_item in sharded_write_items:
      shard = shards[write_item.index.index]
      resolved_data = planner.resolve_data(write_item)
      self.assertTrue(torch.allclose(shard.data, resolved_data))


class DistributedCheckpointHelpersTest(DistributedCheckpointTestBase):

  def test_sharded_cpu_state_dict(self):
    model = self.SimpleLinear().to(xm.xla_device())
    state_dict = model.state_dict()
    sharded_cpu_state_dict = _sharded_cpu_state_dict(state_dict)
    self.assertCountEqual(sharded_cpu_state_dict,
                          ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])
    for name, param in sharded_cpu_state_dict.items():
      if name == 'fc1.weight':
        # _sharded_cpu_state_dict returns _CpuShards only for sharded tensors
        if _is_sharded_tensor(param):
          self.assertTrue(isinstance(param, _CpuShards))
      else:
        self.assertTrue(isinstance(param, torch.Tensor))
        self.assertTrue(param.device == torch.device("cpu"))


class CheckpointManagerTest(DistributedCheckpointTestBase):

  def setUp(self):
    super().setUp()
    # Initialize a minimal process group
    dist.init_process_group(
        backend='gloo',
        init_method='tcp://localhost:8932',
        world_size=1,
        rank=0)
    torch_xla._XLAC._ensure_xla_coordinator_initialized(
        global_rank=0, world_size=1, master_addr="localhost")

  def tearDown(self):
    super().tearDown()
    # Destroy the CPU process group after the test
    dist.destroy_process_group()

  @run_with_tmpdir
  def test_manager_checkpointing(self, tmpdir):
    chkpt_mgr = CheckpointManager(
        tmpdir, save_interval=10, chkpt_on_preemption=False)
    state_dict = self._get_sharded_model().state_dict()

    # Take a checkpoint on step 0
    self.assertTrue(chkpt_mgr.save(0, state_dict))

    # Load the checkpoint into a new state_dict
    new_state_dict = self._get_sharded_model().state_dict()
    self.assertFalse(
        any(
            torch.allclose(v, new_state_dict[k])
            for k, v in state_dict.items()))
    chkpt_mgr.restore(0, new_state_dict)
    self.assertTrue(
        all(
            torch.allclose(v, new_state_dict[k])
            for k, v in state_dict.items()))

  @run_with_tmpdir
  def test_manager_step_tracking(self, tmpdir):
    chkpt_mgr = CheckpointManager(
        tmpdir, save_interval=10, chkpt_on_preemption=False)
    state_dict = self._get_sharded_model().state_dict()

    # No steps are being tracked initially
    self.assertEqual(chkpt_mgr.all_steps(), [])

    # Steps not divisible by 10 should not be saved
    for step in range(1, 10):
      self.assertFalse(chkpt_mgr.save(step, state_dict))
      self.assertEqual(chkpt_mgr.all_steps(), [])

    # Steps divisible by 10 should be saved
    saved = set()
    for step in range(0, 100, 10):
      self.assertTrue(chkpt_mgr.save(step, state_dict))
      saved.add(step)
      self.assertEqual(set(chkpt_mgr.all_steps()), saved)

  @run_with_tmpdir
  def test_manager_max_to_keep(self, tmpdir):
    chkpt_mgr = CheckpointManager(
        tmpdir, save_interval=10, max_to_keep=2, chkpt_on_preemption=False)
    state_dict = self._get_sharded_model().state_dict()

    # No steps are being tracked initially
    self.assertEqual(chkpt_mgr.all_steps(), [])

    self.assertTrue(chkpt_mgr.save(10, state_dict))
    self.assertEqual(set(chkpt_mgr.all_steps()), {10})

    self.assertTrue(chkpt_mgr.save(20, state_dict))
    self.assertEqual(set(chkpt_mgr.all_steps()), {10, 20})

    # The oldest checkpoint should be erased
    self.assertTrue(chkpt_mgr.save(30, state_dict))
    self.assertEqual(set(chkpt_mgr.all_steps()), {30, 20})

    # The oldest is selected by creation timestamp, not step
    self.assertTrue(chkpt_mgr.save(10, state_dict))
    self.assertEqual(set(chkpt_mgr.all_steps()), {30, 10})

    # The deletion order should persist across executions
    chkpt_mgr = CheckpointManager(
        tmpdir, save_interval=10, max_to_keep=2, chkpt_on_preemption=False)
    self.assertTrue(chkpt_mgr.save(20, state_dict))
    self.assertEqual(set(chkpt_mgr.all_steps()), {20, 10})

  @run_with_tmpdir
  def test_manager_async(self, tmpdir):
    chkpt_mgr = CheckpointManager(
        tmpdir, save_interval=10, chkpt_on_preemption=False)
    state_dict = self._get_sharded_model().state_dict()

    # Patch the manager's save method to block until this thread signals.
    cond = threading.Condition()
    old_save = chkpt_mgr._save

    def patched_save(*args, **kwargs):
      with cond:
        cond.wait()
      old_save(*args, **kwargs)

    with unittest.mock.patch.object(chkpt_mgr, '_save', patched_save):
      chkpt_mgr.save_async(10, state_dict)

    # No new steps should be tracked immediately after calling save_async
    self.assertEqual(chkpt_mgr.all_steps(), [])

    # Trigger the actual checkpoint in the background thread and wait for
    # completion.
    with cond:
      cond.notify()
    chkpt_mgr.join()

    # The manager should track all steps which were asynchronously saved.
    self.assertEqual(set(chkpt_mgr.all_steps()), {10})

  @run_with_tmpdir
  def test_manager_async_step_tracking(self, tmpdir):
    chkpt_mgr = CheckpointManager(
        tmpdir, save_interval=10, chkpt_on_preemption=False)
    state_dict = self._get_sharded_model().state_dict()

    self.assertEqual(chkpt_mgr.all_steps(), [])

    # Steps not divisible by 10 should not be saved
    for step in range(1, 10):
      self.assertFalse(chkpt_mgr.save_async(step, state_dict))
      self.assertEqual(chkpt_mgr.all_steps(), [])

    # Steps divisible by 10 should be saved
    saved = set()
    for step in range(0, 100, 10):
      self.assertTrue(chkpt_mgr.save_async(step, state_dict))
      saved.add(step)

    # Join to allow pending async checkpoints to complete
    chkpt_mgr.join()

    # The manager should track all steps which were asynchronously saved.
    self.assertEqual(set(chkpt_mgr.all_steps()), saved)

    # Load a checkpoint into a new state_dict
    new_state_dict = self._get_sharded_model().state_dict()
    self.assertFalse(
        any(
            torch.allclose(v, new_state_dict[k])
            for k, v in state_dict.items()))
    chkpt_mgr.restore(0, new_state_dict)
    self.assertTrue(
        all(
            torch.allclose(v, new_state_dict[k])
            for k, v in state_dict.items()))

  @unittest.skipIf(xr.device_type() != 'TPU',
                   'TPU required for worker IP discovery')
  @unittest.mock.patch('torch_xla._internal.tpu.get_worker_ips')
  def test_master_ip_discovery(self, patched_get_worker_ips):
    # A basic test to verify the SPMD codepath returns the correct IP. Two IPs
    # are needed to avoid the short-circuit return of localhost.
    patched_get_worker_ips.return_value = ['10.0.0.1', '10.0.0.2']
    self.assertTrue(xr.get_master_ip(), '10.0.0.1')

  def test_preemption_sync_manager(self):
    try:
      torch_xla._XLAC._activate_preemption_sync_manager()
      sync_point_reached = torch_xla._XLAC._sync_point_reached

      # No sync point for the first several steps
      sigterm_step = 10
      for step in range(sigterm_step):
        self.assertFalse(sync_point_reached(step))

      # Send a SIGTERM to the current process to trigger a sync point
      os.kill(os.getpid(), signal.SIGTERM)

      # Allow the signal to be processed. The PreemptionSyncManager must receive
      # the SIGTERM, which happens asynchronously, and the state must be
      # propagated through the distributed runtime. Eventually,
      # sync_point_reached will return True.
      success = False
      for attempt in range(10):
        success = sync_point_reached(sigterm_step + attempt)
        if success:
          break
        time.sleep(1)
      self.assertTrue(success, "Sync point was never reached after SIGTERM")
    finally:
      # Scope the PreemptionSyncManager to the lifespan of the test.
      torch_xla._XLAC._deactivate_preemption_sync_manager()

  @unittest.skipIf(xr.device_type() != 'TPU',
                   'TPU required for worker IP discovery')
  @run_with_tmpdir
  def test_auto_checkpoint(self, tmpdir):
    # Create a checkpoint manager with a long save interval
    chkpt_mgr = CheckpointManager(tmpdir, save_interval=100)
    state_dict = self._get_sharded_model().state_dict()

    preemption_step = 10
    # Skip step 0 so the manager will track no checkpoints before preemption
    for step in range(1, preemption_step):
      self.assertFalse(chkpt_mgr.save(step, state_dict))

    with unittest.mock.patch('torch_xla._XLAC._sync_point_reached',
                             lambda x: True):
      self.assertTrue(chkpt_mgr.save(preemption_step, state_dict))
      self.assertTrue(chkpt_mgr.reached_preemption(step))


@unittest.skipIf(xr.device_type() != 'TPU',
                 'TPU required for worker IP discovery')
class OptimizerCheckpointTest(DistributedCheckpointTestBase):

  def setUp(self):
    super().setUp()
    # Initialize a minimal process group
    dist.init_process_group(
        backend='gloo',
        init_method='tcp://localhost:8932',
        world_size=1,
        rank=0)

  def tearDown(self):
    super().tearDown()
    # Destroy the CPU process group after the test
    dist.destroy_process_group()

  def _get_model_and_optimizer(self, optim_cls):
    model = self._get_sharded_model()
    optim = optim_cls(params=model.parameters())
    return model, optim

  def _run_train_step(self, model, optim):
    torch.manual_seed(42)
    model(torch.ones(10, 128).to('xla')).square().sum().backward()
    optim.step()
    xm.mark_step()

  def _test_optimizer(self, tmpdir, optim_cls):
    model, optim = self._get_model_and_optimizer(optim_cls)
    self._run_train_step(model, optim)

    # Take a checkpoint including the optimizer
    chkpt_mgr = CheckpointManager(tmpdir, 1)
    state_dict = {'model': model.state_dict(), 'optim': optim.state_dict()}
    chkpt_mgr.save(0, state_dict, force=True)

    # Load the checkpoint into a new model and optimizer
    new_model, new_optim = self._get_model_and_optimizer(optim_cls)
    prime_optimizer(new_optim)
    new_state_dict = {
        'model': new_model.state_dict(),
        'optim': new_optim.state_dict()
    }
    chkpt_mgr.restore(0, new_state_dict)
    self._assert_same_state_dict(state_dict, new_state_dict)

    new_model.load_state_dict(new_state_dict['model'])
    new_optim.load_state_dict(new_state_dict['optim'])
    self._assert_same_state_dict(new_model.state_dict(), model.state_dict())
    self._assert_same_state_dict(new_optim.state_dict(), optim.state_dict())

  @run_with_tmpdir
  def test_sgd(self, tmpdir):
    self._test_optimizer(tmpdir, torch.optim.SGD)

  @run_with_tmpdir
  def test_adamw(self, tmpdir):
    self._test_optimizer(tmpdir, torch.optim.AdamW)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
