import functools
import os
import sys
import tempfile
import unittest
import test_xla_sharding_base
import threading

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.experimental.xla_sharding as xs

from torch.distributed.checkpoint.default_planner import (
    create_default_local_save_plan,
    create_default_global_save_plan,
)
from torch_xla.experimental.distributed_checkpoint import SPMDLoadPlanner, SPMDSavePlanner, CheckpointManager
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
    xr.use_spmd()
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


class EndToEndCheckpointTest(DistributedCheckpointTestBase):

  def _save_and_restore(self,
                        model_in,
                        model_out,
                        save_planner=None,
                        load_planner=None,
                        is_sharded_cpu_state_dict=False,
                        no_dist=True,
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
    dist_cp.save_state_dict(
        state_dict=model_in_state_dict,
        storage_writer=dist_cp.FileSystemWriter(chkpt_path),
        planner=save_planner,
        no_dist=no_dist,
    )
    # Load the checkpoint using the provided load planner
    for p1, p2 in zip(model_in.parameters(), model_out.parameters()):
      self.assertFalse(torch.allclose(p1.cpu(), p2.cpu()))

    dist_cp.load_state_dict(
        state_dict=model_out_state_dict,
        storage_reader=dist_cp.FileSystemReader(chkpt_path),
        planner=load_planner,
        no_dist=no_dist,
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

  @unittest.skipUnless(
      {'CHKPT_PATH', 'MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE'
      } <= os.environ.keys(),
      'CHKPT_PATH and distributed config must be set for multihost checkpoint')
  def test_multihost_checkpoint(self):
    torch.manual_seed(42)

    # Initialize the default CPU process group from the environment.
    dist.init_process_group()

    model1 = self._get_sharded_model(mesh_shape=(1, self.n_devices))
    model2 = self._get_sharded_model(mesh_shape=(self.n_devices, 1))
    # Take the checkpoint, writing to the path configured in the environment.
    self._save_and_restore(
        model1,
        model2,
        save_planner=SPMDSavePlanner(),
        load_planner=SPMDLoadPlanner(),
        no_dist=False,
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
    # Initialize the a minimal process group
    dist.init_process_group(
        backend='gloo', init_method='tcp://127.1:8932', world_size=1, rank=0)

  def tearDown(self):
    super().tearDown()
    # Destroy the CPU process group after the test
    dist.destroy_process_group()

  @run_with_tmpdir
  def test_manager_checkpointing(self, tmpdir):
    chkpt_mgr = CheckpointManager(tmpdir, save_interval=10)
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
    chkpt_mgr = CheckpointManager(tmpdir, save_interval=10)
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
    chkpt_mgr = CheckpointManager(tmpdir, save_interval=10, max_to_keep=2)
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
    chkpt_mgr = CheckpointManager(tmpdir, save_interval=10, max_to_keep=2)
    self.assertTrue(chkpt_mgr.save(20, state_dict))
    self.assertEqual(set(chkpt_mgr.all_steps()), {20, 10})

  @run_with_tmpdir
  def test_manager_async(self, tmpdir):
    chkpt_mgr = CheckpointManager(tmpdir, save_interval=10)
    state_dict = self._get_sharded_model().state_dict()

    # Patch the manager's save method to block until this thread signals.
    cond = threading.Condition()
    old_save = chkpt_mgr.save

    def patched_save(*args, **kwargs):
      cond.wait()
      old_save(*args, **kwargs)

    with unittest.mock.patch.object(chkpt_mgr, 'save', patched_save):
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
    chkpt_mgr = CheckpointManager(tmpdir, save_interval=10)
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


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
