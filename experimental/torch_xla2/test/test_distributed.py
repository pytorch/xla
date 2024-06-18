import os
import jax
import numpy as np

import pytest
import torch.distributed as dist
import torch_xla2
import torch_xla2.distributed


@pytest.fixture(scope='module')
def multi_cpu():
  # TODO(wcromar): support other devices
  jax.config.update('jax_platforms', 'cpu')
  replicas = 4
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={replicas}"

  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  dist.init_process_group(backend="jax", init_method="jax://")

  yield jax.device_count()
  dist.destroy_process_group()


@pytest.mark.parametrize(('op', 'expected'), [
  (dist.ReduceOp.SUM, sum(range(4))),
  (dist.ReduceOp.AVG, sum(range(4)) / 4),
  (dist.ReduceOp.MIN, 0),
  (dist.ReduceOp.MAX, 3),
])
def test_all_reduce(op, expected, multi_cpu):
  device_count = multi_cpu

  def f(index):
    dist.all_reduce(index, op)
    return index

  res = torch_xla2.distributed.spawn(f)

  expected_tensors = [expected for _ in range(device_count)]
  np.testing.assert_equal(res.numpy(), expected_tensors)
