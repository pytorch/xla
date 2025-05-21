import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils._pytree as pytree
from absl.testing import absltest, parameterized
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.debug.metrics as met
from torch_xla._internal import pjrt, tpu


# Test for collective ops from xla_model
class TestXMCollectiveOpsTpu(parameterized.TestCase):

  @staticmethod
  def _broadcast(sync):
    torch.manual_seed(xr.global_ordinal())
    device = xm.xla_device()
    model = nn.Linear(5, 5).to(device)
    if sync:
      xm.broadcast_master_param(model)

    torch_xla.sync()
    return next(model.parameters()).detach().cpu().numpy()

  @absltest.skipUnless(tpu.num_tpu_workers() == 1,
                       "Only implemented for single host.")
  @parameterized.named_parameters(('synchronized_parameters', True),
                                  ('unsynchronized_parameters', False))
  def test_broadcast_master_param(self, sync):
    results = pjrt.run_multiprocess(self._broadcast, sync)
    master_params = results[0]
    for ordinal, worker_params in results.items():
      if sync:
        np.testing.assert_array_equal(master_params, worker_params)
      elif ordinal != 0:
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 master_params, worker_params)

  @staticmethod
  def _all_reduce(pin_layout):
    device = xm.xla_device()
    # Prevent 0 and 1 from being converted to constants
    ordinal = xm.send_cpu_data_to_device(
        torch.tensor(
            xr.global_ordinal(), dtype=torch.float32, requires_grad=True),
        device=device)
    out = xm.all_reduce(xm.REDUCE_SUM, ordinal, pin_layout=pin_layout)[0]
    assert out.requires_grad
    torch_xla.sync()

    return out.cpu().detach().numpy()

  @parameterized.named_parameters(('pinned', True), ('unpinned', False))
  def test_all_reduce(self, pin_layout):
    results = pjrt.run_multiprocess(self._all_reduce, pin_layout)

    expected = sum(range(tpu.num_expected_global_devices()))
    for v in results.values():
      np.testing.assert_array_equal(v, expected)

  @staticmethod
  def _all_gather(pin_layout):
    device = xm.xla_device()
    ordinal = torch.tensor([xr.global_ordinal()], device=device)
    out = xm.all_gather(ordinal, pin_layout=pin_layout)
    torch_xla.sync()

    return out.cpu().numpy()

  @parameterized.named_parameters(('pinned', True), ('unpinned', False))
  def test_all_gather(self, pin_layout):
    results = pjrt.run_multiprocess(self._all_gather, pin_layout)

    expected = list(range(tpu.num_expected_global_devices()))
    for v in results.values():
      np.testing.assert_array_equal(v, expected)

  @staticmethod
  def _reduce_scatter(pin_layout):
    device = xm.xla_device()
    world_size = xr.world_size()
    tensor = -torch.arange(world_size, dtype=torch.float32).to(device)

    out = xm.reduce_scatter(
        xm.REDUCE_SUM,
        tensor,
        scale=1.0 / world_size,
        scatter_dim=0,
        shard_count=world_size,
        pin_layout=pin_layout,
    )
    torch_xla.sync()

    return out.cpu().numpy()

  @parameterized.named_parameters(('pinned', True), ('unpinned', False))
  def test_reduce_scatter(self, pin_layout):
    results = pjrt.run_multiprocess(self._reduce_scatter, pin_layout)

    for ordinal, value in results.items():
      np.testing.assert_array_equal(value, [-ordinal])

  @staticmethod
  def _all_to_all(pin_layout):
    device = xm.xla_device()
    world_size = xr.world_size()

    tensor = torch.cat(
        [
            -torch.arange(world_size, dtype=torch.float32).view(-1, 1, 1),
            torch.ones(world_size, 1, 1) * xr.global_ordinal(),
        ],
        dim=1,
    ).to(device)
    torch_xla.sync()

    out = xm.all_to_all(
        tensor,
        split_dimension=0,
        concat_dimension=2,
        split_count=world_size,
        pin_layout=pin_layout,
    )

    return out.cpu().numpy()

  @parameterized.named_parameters(('pinned', True), ('unpinned', False))
  def test_all_to_all(self, pin_layout):
    results = pjrt.run_multiprocess(self._all_to_all, pin_layout)

    world_size = tpu.num_expected_global_devices()
    for ordinal, value in results.items():
      np.testing.assert_array_equal(value, [[[-ordinal] * world_size,
                                             list(range(world_size))]])


@absltest.skipIf(tpu.num_logical_cores_per_chip() >= 2,
                 "Dynamo not supported on TPU v2/v3")
class TestDistCollectiveOpsTpu(parameterized.TestCase):
  """Test for collective ops from torch.distributed"""

  @staticmethod
  def _all_reduce(use_dynamo: bool):
    met.clear_all()

    def callable(input):
      dist.all_reduce(input, dist.ReduceOp.SUM)
      return input

    dist.init_process_group("xla", init_method='xla://')
    device = xm.xla_device()
    input = torch.tensor([xr.global_ordinal()],
                         dtype=torch.float,
                         device=device)

    f = torch.compile(callable, backend='openxla') if use_dynamo else callable
    f(input)
    torch_xla.sync()
    if not use_dynamo:
      assert 'xla::AllReduceInPlace' in met.counter_names(
      ) or 'xla::AllReduce' in met.counter_names()
    else:
      assert 'xla::all_reduce' in met.counter_names()
    return input.cpu()

  @staticmethod
  def _all_gather_into_tensor(use_dynamo: bool):
    met.clear_all()

    def callable(output, input):
      dist.all_gather_into_tensor(output_tensor, input, None)
      return output_tensor

    dist.init_process_group("xla", init_method='xla://')
    device = xm.xla_device()
    input = torch.tensor([xr.global_ordinal()],
                         dtype=torch.float,
                         device=device)
    output_tensor = torch.empty((1, xr.world_size()), device=device)
    f = torch.compile(callable, backend='openxla') if use_dynamo else callable
    f(output_tensor, input)
    torch_xla.sync()
    if not use_dynamo:
      assert 'xla::AllGather' in met.counter_names(
      ) or 'xla::AllGatherOut' in met.counter_names()
    else:
      assert 'xla::all_gather_into_tensor' in met.counter_names()
    return output_tensor.cpu()

  @staticmethod
  def _all_gather(use_dynamo: bool):
    met.clear_all()
    dist.init_process_group("xla", init_method='xla://')
    device = xm.xla_device()

    def callable(input):
      output_tensor = [
          torch.tensor([0], dtype=torch.float).to(device)
          for _ in range(xr.world_size())
      ]
      dist.all_gather(output_tensor, input, None)
      return output_tensor

    input = torch.tensor([xr.global_ordinal()],
                         dtype=torch.float,
                         device=device)

    f = torch.compile(callable, backend='openxla') if use_dynamo else callable
    output = f(input)
    torch_xla.sync()
    if not use_dynamo:
      assert 'xla::AllGather' in met.counter_names(
      ) or 'xla::AllGatherOut' in met.counter_names()
    else:
      assert 'xla::all_gather_into_tensor' in met.counter_names()
    # output is list of tensors
    return pytree.tree_map(lambda x: x.cpu(), output)

  @staticmethod
  def _reduce_scatter(use_dynamo: bool):
    met.clear_all()
    dist.init_process_group("xla", init_method='xla://')
    device = xm.xla_device()

    def callable(output, input):
      dist.reduce_scatter_tensor(output, input)
      return output

    # check https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/torch/distributed/distributed_c10d.py#L3766-L3814
    # for input and output tensor example
    tensor_in = torch.arange(
        xr.world_size() * 2, dtype=torch.float, device=device)
    tensor_out = torch.zeros(2, dtype=torch.float, device=device)
    f = torch.compile(callable, backend='openxla') if use_dynamo else callable
    output = f(tensor_out, tensor_in)
    torch_xla.sync()
    if not use_dynamo:
      assert 'xla::ReduceScatter' in met.counter_names(
      ) or 'xla::ReduceScatterOut' in met.counter_names()
    else:
      assert 'xla::reduce_scatter_tensor' in met.counter_names()
    return output.cpu()

  @staticmethod
  def _all_to_all_single(use_dynamo: bool, split_size: int = 1):
    met.clear_all()
    dist.init_process_group("xla", init_method='xla://')
    device = xm.xla_device()

    def callable(output, input):
      dist.all_to_all_single(output, input)
      return output

    # check https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/torch/distributed/distributed_c10d.py#L3880
    # for input and output tensor example
    tensor_in = torch.tensor(
        [xr.local_ordinal()] * (tpu.num_expected_global_devices() * split_size),
        dtype=torch.float,
        device=device)
    tensor_out = torch.zeros_like(tensor_in)
    f = torch.compile(callable, backend='openxla') if use_dynamo else callable
    output = f(tensor_out, tensor_in)
    torch_xla.sync()
    if not use_dynamo:
      assert 'xla::AllToAll' in met.counter_names()
    else:
      assert 'xla::all_to_all_single' in met.counter_names()
    return output.cpu()

  @parameterized.named_parameters(('dynamo', True), ('nondynamo', False))
  def test_all_reduce(self, use_dynamo):
    results = pjrt.run_multiprocess(self._all_reduce, use_dynamo=use_dynamo)
    expected = torch.tensor([sum(range(tpu.num_expected_global_devices()))],
                            dtype=torch.float)
    for index, val in results.items():
      torch.testing.assert_close(val, expected)

  @parameterized.named_parameters(('dynamo', True), ('nondynamo', False))
  def test_all_gather_into_tensor(self, use_dynamo):
    results = pjrt.run_multiprocess(
        self._all_gather_into_tensor, use_dynamo=use_dynamo)
    expected = torch.arange(
        tpu.num_expected_global_devices(), dtype=torch.float).unsqueeze(0)
    for index, val in results.items():
      torch.testing.assert_close(val, expected)

  @parameterized.named_parameters(('dynamo', True), ('nondynamo', False))
  def test_all_gather(self, use_dynamo):
    results = pjrt.run_multiprocess(self._all_gather, use_dynamo=use_dynamo)
    expected = [
        torch.tensor([i], dtype=torch.float)
        for i in range(tpu.num_expected_global_devices())
    ]
    for index, val in results.items():
      torch.testing.assert_close(val, expected)

  @parameterized.named_parameters(('dynamo', True), ('nondynamo', False))
  def test_reduce_scatter(self, use_dynamo):
    results = pjrt.run_multiprocess(self._reduce_scatter, use_dynamo=use_dynamo)
    expected = [
        torch.tensor([
            2 * i * tpu.num_expected_global_devices(),
            (2 * i + 1) * tpu.num_expected_global_devices()
        ],
                     dtype=torch.float)
        for i in range(tpu.num_expected_global_devices())
    ]
    for index, val in results.items():
      torch.testing.assert_close(val, expected[index])

  @parameterized.named_parameters(('dynamo', True), ('nondynamo', False))
  def test_all_to_all_single(self, use_dynamo):
    split_size = 2
    results = pjrt.run_multiprocess(
        self._all_to_all_single, use_dynamo=use_dynamo, split_size=split_size)
    expected = torch.arange(
        tpu.num_expected_global_devices(), dtype=torch.float).repeat(split_size)
    # Note: AllToAll xla op does not honor the order of the all_to_all, which means
    # the rank may not follow the order.
    for _, val in results.items():
      self.assertTrue(
          torch.allclose(val.sort().values,
                         expected.sort().values),
          f"Got {val}, expected {expected}")


if __name__ == '__main__':
  absltest.main()
