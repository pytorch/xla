import os
import sys
import tempfile
import unittest
import test_xla_sharding_base

import torch
import torch.distributed.checkpoint as dist_cp
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.experimental.xla_sharding as xs

from torch.distributed.checkpoint.default_planner import (
    create_default_local_save_plan,
    create_default_global_save_plan,
)
from torch_xla.experimental.distributed_checkpoint import SPMDLoadPlanner, SPMDSavePlanner
from torch_xla.experimental._distributed_checkpoint_helpers import (
    _sharded_cpu_state_dict, _CpuShards, _is_sharded_tensor)


class DistributedCheckpointTestBase(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    os.environ['XLA_USE_SPMD'] = '1'
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


class ReshardingTest(DistributedCheckpointTestBase):

  def _save_and_restore(self,
                        model_in,
                        model_out,
                        save_planner=None,
                        load_planner=None):
    """
    Checkpoint model_in using the provided save_planner and load into model_out
    using the provided load_planner, and assert model_out equals model_in after
    the load. If either planner is not specified, the DefaultPlanner is used.
    """
    tmpdir = tempfile.mkdtemp()

    # Save an unsharded model using the provided save planner
    dist_cp.save_state_dict(
        state_dict=model_in.state_dict(),
        storage_writer=dist_cp.FileSystemWriter(tmpdir),
        planner=save_planner,
        no_dist=True,  # Single-host checkpoint doesn't require a process group
    )

    # Load the checkpoint using the provided load planner
    for p1, p2 in zip(model_in.parameters(), model_out.parameters()):
      self.assertFalse(torch.allclose(p1, p2))
    dist_cp.load_state_dict(
        state_dict=model_out.state_dict(),
        storage_reader=dist_cp.FileSystemReader(tmpdir),
        planner=load_planner,
        no_dist=True,  # Single-host checkpoint doesn't require a process group
    )
    for p1, p2 in zip(model_in.parameters(), model_out.parameters()):
      self.assertTrue(torch.allclose(p1, p2))

  def test_unsharded_to_sharded(self):
    # Save an unsharded model using the DefaultSavePlanner and load into a
    # sharded model using the SPMDLoadPlanner
    model = self.SimpleLinear().to(xm.xla_device())
    sharded_model = self._get_sharded_model()
    self._save_and_restore(model, sharded_model, load_planner=SPMDLoadPlanner())

  # TODO(jonbolin): Enable tests for resharding into coarser meshes
  @unittest.skip("View assignment with virtual device is not yet supported")
  def test_sharded_to_unsharded(self):
    model = self.SimpleLinear().to(xm.xla_device())
    sharded_model = self._get_sharded_model()
    self._save_and_restore(sharded_model, model, save_planner=SPMDSavePlanner())

  # TODO(jonbolin): Enable tests for resharding into coarser meshes
  @unittest.skip("View assignment with virtual device is not yet supported")
  @unittest.skipIf(xr.global_device_count() == 1,
                   "Multiple devices needed to change mesh")
  def test_different_device_mesh(self):
    dim = self.n_devices // 2
    model1 = self._get_sharded_model(mesh_shape=(dim, self.n_devices // dim))
    model2 = self._get_sharded_model(mesh_shape=(self.n_devices, 1))
    self._save_and_restore(
        model1,
        model2,
        save_planner=SPMDSavePlanner(),
        load_planner=SPMDLoadPlanner())


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

  @unittest.skipIf(xr.global_device_count() == 1,
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

  def _get_save_planner(self, model):
    # Create an SPMDSavePlanner for the given model.
    planner = SPMDSavePlanner()
    planner.set_up_planner(model.state_dict(), True)
    return planner

  def test_state_dict_separation(self):
    model = self._get_sharded_model()
    planner = self._get_save_planner(model)
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

  def test_local_save_plan(self):
    model = self._get_sharded_model()
    planner = self._get_save_planner(model)
    plan = planner.create_local_plan()
    parameter_count = len(list(model.parameters()))
    if self.n_devices > 1:
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
      # If unsharded, there should be a single WriteItem per model parameter
      self.assertEqual(parameter_count, len(plan.items))

  @unittest.skipIf(xr.global_device_count() == 1,
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


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
