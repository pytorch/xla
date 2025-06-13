import os
import jax
import numpy as np

import pytest
import torch
import torch.distributed._functional_collectives
import torch.distributed as dist
import torchax
import torchax.distributed

# Dummy group name to use with functional collectives. Ignored by
# implementations.
# TODO(wcromar): do something useful with group name
GROUP_NAME = "process_group"

torchax.enable_globally()


@pytest.fixture(scope="module")
def multi_cpu():
  # TODO(wcromar): support other device counts
  assert (jax.device_count() == 4
         ), "Set XLA_FLAGS=--xla_force_host_platform_device_count=4 if on CPU"

  yield jax.device_count()


@pytest.fixture()
def process_group():
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  dist.init_process_group(backend="jax", init_method="jax://")
  # HACK: our default process group has world size 1, regardless of actual
  # device count. Only put rank 0 so PyTorch doesn't complain about non-existent
  # ranks. Our lowerings ignore this list, so this ends up being fine.
  # TODO(wcromar): Figure out if there's a cleaner way
  group_ranks = [0]
  yield group_ranks
  dist.destroy_process_group()


def test_all_gather_tensor(multi_cpu, process_group):
  device_count = multi_cpu

  def f(index: torchax.tensor.Tensor):
    with torchax.default_env():
      output = torch.zeros_like(index).expand(device_count)
    dist.all_gather_into_tensor(output, index)
    return output

  res = torchax.distributed.spawn(f)

  expected_tensors = [[0, 1, 2, 3] for _ in range(device_count)]
  np.testing.assert_equal([r.numpy() for r in res], expected_tensors)


def test_all_gather_tensor_func(multi_cpu, process_group):
  device_count = multi_cpu
  group_ranks = process_group

  def f(index: torchax.tensor.Tensor):
    return torch.distributed._functional_collectives.all_gather_tensor(
        index, 0, group_ranks)

  res = torchax.distributed.spawn(f)

  expected_tensors = [[0, 1, 2, 3] for _ in range(device_count)]
  np.testing.assert_equal([r.numpy() for r in res], expected_tensors)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (dist.ReduceOp.SUM, sum(range(4))),
        (dist.ReduceOp.AVG, sum(range(4)) // 4),
        (dist.ReduceOp.MIN, 0),
        (dist.ReduceOp.MAX, 3),
    ],
)
def test_all_reduce(op, expected, multi_cpu, process_group):
  device_count = multi_cpu

  def f(index):
    with torchax.default_env():
      dist.all_reduce(index, op)
      return index

  res = torchax.distributed.spawn(f)

  expected_tensors = [expected for _ in range(device_count)]
  np.testing.assert_equal(res.numpy(), expected_tensors)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("sum", sum(range(4))),
        ("avg", sum(range(4)) / 4),
        ("min", 0),
        ("max", 3),
    ],
)
def test_all_reduce_func(op, expected, multi_cpu):
  device_count = multi_cpu

  def f(index):
    return torch.distributed._functional_collectives.all_reduce(
        index, op, GROUP_NAME)

  res = torchax.distributed.spawn(f)

  expected_tensors = [expected for _ in range(device_count)]
  np.testing.assert_equal(res.numpy(), expected_tensors)


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, 0),
        (2, 2),
    ],
)
def test_broadcast(rank, expected, multi_cpu, process_group):
  device_count = multi_cpu

  def f(index):
    dist.broadcast(index, rank)
    return index

  res = torchax.distributed.spawn(f)

  expected_tensors = [expected for _ in range(device_count)]
  np.testing.assert_equal(res.numpy(), expected_tensors)


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, 0),
        (2, 2),
    ],
)
def test_broadcast_func(rank, expected, multi_cpu):
  device_count = multi_cpu

  def f(index):
    return torch.distributed._functional_collectives.broadcast(
        index, rank, GROUP_NAME)

  res = torchax.distributed.spawn(f)

  expected_tensors = [expected for _ in range(device_count)]
  np.testing.assert_equal(res.numpy(), expected_tensors)
