from absl.testing import absltest
from copy import deepcopy
import time
import unittest
import functools

import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr
import torch_xla.experimental.assume_pure as ap


# TODO: Merge assume_pure_torch and assume_pure tests after implementing
# backward pass for assume_pure_torch.
class TestAssumePure(absltest.TestCase):

  def setUp(self):
    super().setUp()
    torch.set_grad_enabled(False)
    ap._XLA_COMPUTATION_CACHE.clear()

  def test_assume_pure_basic(self):
    # Arrange
    @ap.assume_pure_torch
    def simple_torch_function(a, b):
      result = torch.sin(a @ b)
      return result

    # Act
    a = torch.ones((3, 3), device='xla', requires_grad=True)
    actual = simple_torch_function(a, a)
    torch_xla.sync()

    # Assert
    expected = torch.sin(torch.ones(3, 3) @ torch.ones(3, 3))
    torch.testing.assert_close(actual, expected, check_device=False)

  @unittest.skipUnless(xr.global_runtime_device_count() >= 2,
                       "Multiple devices required")
  def test_assume_pure_other_xla_devices(self):
    # Preconditions: ensure we have at least two XLA devices.
    assert torch.device('xla:0') != torch.device('xla:1')

    # Arrange
    @ap.assume_pure_torch
    def simple_torch_function(a, b):
      return torch.sin(a @ b)

    # Act: use an XLA device with ID 1.
    a = torch.ones((3, 3), device='xla:1', requires_grad=True)
    actual = simple_torch_function(a, a)
    torch_xla.sync()

    # Assert
    expected = torch.sin(torch.ones(3, 3) @ torch.ones(3, 3))
    torch.testing.assert_close(actual, expected, check_device=False)

  def test_assume_pure_module(self):
    # Arrange
    model = nn.Linear(3, 3).to('xla')

    @ap.assume_pure_torch
    def simple_torch_function(params, x):
      return torch.func.functional_call(model, params, x)

    # Act
    a = torch.ones((3, 3), device='xla', requires_grad=True)
    model_params = dict(model.named_parameters())
    actual = simple_torch_function(model_params, a)
    torch_xla.sync()

    # Assert
    expected = model(torch.ones(3, 3).to('xla'))
    torch.testing.assert_close(actual, expected, check_device=False)

  def test_assume_pure_complex_module(self):
    """Test a module comprising of some linear, conv, and relu layers."""

    # Arrange: define module and prepare inputs.
    class MyModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = nn.Linear(9, 9)
        self.conv = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=(1, 1))
        self.layer_norm = nn.LayerNorm(9)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(9 * 9 * 3, 3)

      def forward(self, x):
        x = self.linear(x)
        x = self.conv(x)
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    orig_model = MyModule()
    pure_model = deepcopy(orig_model)
    orig_model = orig_model.to('xla')
    pure_model = pure_model.to('xla')
    pure_params = dict(pure_model.named_parameters())
    orig_x = torch.randn((5, 3, 9, 9), device='xla', requires_grad=True)
    pure_x = orig_x.clone().detach().requires_grad_(True)
    torch_xla.sync()

    # Act: call module in a pure way.
    orig_output = orig_model(orig_x)

    @ap.assume_pure_torch
    def pure_call(params, x):
      return torch.func.functional_call(pure_model, params, x)

    pure_output = pure_call(pure_params, pure_x)
    torch_xla.sync()

    # Assert
    # Check that the outputs are close
    torch.testing.assert_close(pure_output, orig_output, check_device=False)

  def test_assume_pure_avoid_retracing_avoid_rejit(self):
    """Tests that we avoid retracing and re-jitting when using assume_pure."""

    # Arrange
    trace_counter = 0

    @functools.partial(ap.assume_pure_torch, use_cache=True)
    def simple_torch_function(a, b):
      nonlocal trace_counter
      trace_counter += 1
      return torch.sin(a @ b)

    # Act: simulate a training loop.
    for _ in range(5):
      a = torch.ones((3, 3), device='xla', requires_grad=True)
      simple_torch_function(a, a)
      torch_xla.sync()

    # Assert
    # Check that we only trace once.
    self.assertEqual(trace_counter, 1)

  def test_assume_pure_recursive(self):

    @ap.assume_pure_torch
    def torch_func(a, b):
      return torch.matmul(a, b)

    @ap.assume_pure_torch
    def f(a, b):
      y = torch_func(a, b)
      return y + 1

    a = torch.randn(3, 3, device='xla')
    b = torch.randn(3, 3, device='xla')

    output_pure = f(a, b)
    torch.testing.assert_close(
        output_pure,
        a @ b + 1,
        msg="Forward outputs do not match",
        check_device=False)


class TracingBenchmark(absltest.TestCase):

  def setUp(self):
    super().setUp()
    torch.set_grad_enabled(False)
    ap._XLA_COMPUTATION_CACHE.clear()

  def test_trace_transformer_with_spda_attention(self):
    num_iterations = 3
    print(f"\nRunning benchmark with {num_iterations} iterations")

    import sys
    import os
    example_folder = os.path.dirname(os.path.dirname(__file__)) + "/examples"
    sys.path.append(example_folder)
    from decoder_only_model import DecoderOnlyConfig, DecoderOnlyModel  # type:ignore

    config = DecoderOnlyConfig(
        hidden_size=128,
        num_hidden_layers=100,
        intermediate_size=8 * 128,
        vocab_size=256)
    model = DecoderOnlyModel(config=config).to('xla')
    batch_size = 2
    sequence_length = 8

    # Generate random input_ids within the range of the vocabulary size
    input_ids = torch.randint(0, config.vocab_size,
                              (batch_size, sequence_length)).to('xla')

    pure_model = deepcopy(model)
    torch_xla.sync()

    # Test tracing the model normally.
    model(input_ids)  # Warm up
    start_time = time.time()
    for _ in range(num_iterations):
      model(input_ids)
    end_time = time.time()
    model_time = (end_time - start_time) / num_iterations
    print(f"No `@assume_pure` time: {model_time * 1000:.4f} ms")

    # Test tracing the model with assume_pure.
    @functools.partial(ap.assume_pure_torch, use_cache=True)
    def pure_call(params, x):
      return torch.func.functional_call(pure_model, params, x)

    params = dict(pure_model.named_parameters())
    pure_call(params, input_ids)  # Warm up
    start_time = time.time()
    for _ in range(num_iterations):
      pure_call(params, input_ids)
    end_time = time.time()
    pure_model_time = (end_time - start_time) / num_iterations
    print(f"`@assume_pure` time: {pure_model_time * 1000:.4f} ms")


if __name__ == "__main__":
  torch.set_default_dtype(torch.float32)
  torch.manual_seed(42)
  torch_xla.manual_seed(42)
  absltest.main()
