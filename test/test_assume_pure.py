from copy import deepcopy
from absl.testing import absltest
from absl import flags
import time
import unittest

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_builder as xb
import torch_xla.runtime as xr
from torch_xla.experimental.assume_pure import assume_pure
from torch_xla._internal.jax_workarounds import jax_import_guard


def assert_gradients_close(test_case, actual, expected):
  """Checks that the gradients of the `actual` tensor is close to the gradients of the `expected` tensor."""

  grad1 = actual.grad
  grad2 = expected.grad
  if grad1 is None and grad2 is None:
    test_case.fail("Both gradients are None, which is unexpected")
  elif grad1 is None or grad2 is None:
    test_case.fail(
        f"Gradient mismatch: one is None, the other is not. Grad1: {grad1}, Grad2: {grad2}"
    )
  else:
    torch.testing.assert_close(
        grad1.detach(),
        grad2.detach(),
        msg=lambda s: f"Gradients do not match {s}",
        check_device=False)


class TestAssumePure(absltest.TestCase):

  def test_assume_pure_basic(self):
    # Arrange
    @assume_pure
    def simple_torch_function(a, b):
      return torch.sin(a @ b)

    # Act
    a = torch.ones((3, 3), device='xla', requires_grad=True)
    actual = simple_torch_function(a, a)
    actual.sum().backward()
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
    @assume_pure
    def simple_torch_function(a, b):
      return torch.sin(a @ b)

    # Act: use an XLA device with ID 1.
    a = torch.ones((3, 3), device='xla:1', requires_grad=True)
    actual = simple_torch_function(a, a)
    actual.sum().backward()
    torch_xla.sync()

    # Assert
    expected = torch.sin(torch.ones(3, 3) @ torch.ones(3, 3))
    torch.testing.assert_close(actual, expected, check_device=False)

  def test_assume_pure_module(self):
    # Arrange
    model = nn.Linear(3, 3).to('xla')

    @assume_pure
    def simple_torch_function(params, x):
      return torch.func.functional_call(model, params, x)

    # Act
    a = torch.ones((3, 3), device='xla', requires_grad=True)
    actual = simple_torch_function(dict(model.named_parameters()), a)
    actual.sum().backward()
    torch_xla.sync()

    # Assert
    expected = model(torch.ones(3, 3).to('xla'))
    torch.testing.assert_close(actual, expected, check_device=False)

  @unittest.skipIf(xr.device_type() == 'TPU',
                   "Bug: https://github.com/pytorch/xla/issues/8974")
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
    orig_params = dict(orig_model.named_parameters())
    pure_params = dict(pure_model.named_parameters())
    orig_x = torch.randn((5, 3, 9, 9), device='xla', requires_grad=True)
    pure_x = orig_x.clone().detach().requires_grad_(True)
    torch_xla.sync()

    # Act: call module in a pure way.
    orig_output = orig_model(orig_x)
    pure_call = lambda params, x: torch.func.functional_call(
        pure_model, params, x)
    pure_output = assume_pure(pure_call)(pure_params, pure_x)
    torch_xla.sync()

    # Assert
    # Check that the outputs are close
    torch.testing.assert_close(pure_output, orig_output, check_device=False)

    # Check that the gradients are close
    orig_output.sum().backward()
    pure_output.sum().backward()
    torch_xla.sync()
    assert_gradients_close(self, pure_x, orig_x)
    for name, _ in orig_model.named_parameters():
      orig_param = orig_params[name]
      pure_param = pure_params[name]
      assert_gradients_close(self, orig_param, pure_param)

  def test_assume_pure_avoid_retracing_avoid_rejit(self):
    """Tests that we avoid retracing and re-jitting when using assume_pure."""

    # Arrange: first clear the cache to prevent contamination from other tests.
    xb._JAX_TO_XLA_COMPUTATION_CACHE.clear()
    starting_lowerings = xb._jax_to_xla_computation_cache_elements()
    trace_counter = 0

    @assume_pure
    def simple_torch_function(a, b):
      nonlocal trace_counter
      trace_counter += 1
      return torch.sin(a @ b)

    # Act: simulate a training loop.
    for _ in range(5):
      a = torch.ones((3, 3), device='xla', requires_grad=True)
      o = simple_torch_function(a, a)
      o.sum().backward()
      torch_xla.sync()

    # Assert
    ending_lowerings = xb._jax_to_xla_computation_cache_elements()

    # Check that we only trace once.
    self.assertEqual(trace_counter, 1)

    # Check that we only lower to HLO twice (once for forward, once for backward).
    self.assertEqual(ending_lowerings - starting_lowerings, 2)

  def test_assume_pure_matmul_grads(self):
    """Tests matmul with all inputs requiring gradients."""

    # Arrange
    def matmul_fn(a, b):
      return a @ b

    # Prepare inputs (cloned for independent grad computation)
    a_orig = torch.randn(4, 5, device='xla', requires_grad=True)
    b_orig = torch.randn(5, 3, device='xla', requires_grad=True)
    a_pure = a_orig.clone().detach().requires_grad_(True)
    b_pure = b_orig.clone().detach().requires_grad_(True)

    # Act
    # Forward pass
    output_orig = matmul_fn(a_orig, b_orig)
    output_pure = assume_pure(matmul_fn)(a_pure, b_pure)

    # Backward pass
    loss_orig = output_orig.sum()
    loss_pure = output_pure.sum()

    loss_orig.backward()
    loss_pure.backward()
    torch_xla.sync()

    # Assert
    # Check forward pass equivalence
    torch.testing.assert_close(
        output_pure,
        output_orig,
        msg="Forward outputs do not match",
        check_device=False)

    # Check gradients
    assert_gradients_close(self, a_pure, a_orig)
    assert_gradients_close(self, b_pure, b_orig)

  @unittest.skipIf(xr.device_type() == 'TPU',
                   "Bug: https://github.com/pytorch/xla/issues/8975")
  def test_assume_pure_einsum_grads(self):
    """Tests einsum with all inputs requiring gradients."""

    # Arrange
    def einsum_fn(x, y):
      return torch.einsum('bij,bjk->bik', x, y)

    # Prepare inputs
    x_orig = torch.randn(2, 3, 4, device='xla', requires_grad=True)
    y_orig = torch.randn(2, 4, 5, device='xla', requires_grad=True)
    x_pure = x_orig.clone().detach().requires_grad_(True)
    y_pure = y_orig.clone().detach().requires_grad_(True)

    # Act
    # Forward pass
    output_orig = einsum_fn(x_orig, y_orig)
    output_pure = assume_pure(einsum_fn)(x_pure, y_pure)
    torch.testing.assert_close(
        output_pure,
        output_orig,
        msg=lambda msg: f"Forward outputs do not match: {msg}",
        check_device=False)

    # Backward pass
    output_orig.sum().backward()
    output_pure.sum().backward()
    torch_xla.sync()

    # Assert
    # Check gradients
    assert_gradients_close(self, x_pure, x_orig)
    assert_gradients_close(self, y_pure, y_orig)

  def test_assume_pure_partial_grads_args(self):
    """Tests a function where only some positional inputs require gradients.
    
    In this test, tensor a, c require grad; b does not.
    """

    # Arrange
    def fn(a, b, c):
      return a * torch.tanh(b) + c**2

    # Prepare inputs
    torch_xla.manual_seed(42)
    a_orig = torch.randn(3, 3, device='xla', requires_grad=True)
    # No grad for b
    b_orig = torch.randn(3, 3, device='xla', requires_grad=False)
    c_orig = torch.randn(3, 3, device='xla', requires_grad=True)

    a_pure = a_orig.clone().detach().requires_grad_(True)
    # No grad for b
    b_pure = b_orig.clone().detach().requires_grad_(False)
    c_pure = c_orig.clone().detach().requires_grad_(True)

    # Act
    # Forward pass
    output_orig = fn(a_orig, b_orig, c_orig)
    output_pure = assume_pure(fn)(a_pure, b_pure, c_pure)
    torch.testing.assert_close(
        output_pure,
        output_orig,
        msg="Forward outputs do not match",
        check_device=False)

    # Backward pass
    output_orig.sum().backward()
    output_pure.sum().backward()
    torch_xla.sync()

    # Assert
    # Check gradients
    assert_gradients_close(self, a_pure, a_orig)
    assert_gradients_close(self, c_pure, c_orig)

    self.assertIsNotNone(a_orig.grad, "a_orig should have grad")
    self.assertIsNone(b_orig.grad, "b_orig should not have grad")
    self.assertIsNone(b_pure.grad, "b_pure should not have grad")
    self.assertIsNotNone(c_orig.grad, "a_orig should have grad")

  def test_assume_pure_partial_grads_kwargs(self):
    """Tests a function where inputs requiring gradients are passed via kwargs."""

    # Arrange
    def fn(x, *, factor, bias):
      # x, bias require grad; factor does not
      # factor is a non-tensor kwarg, bias is a tensor kwarg
      return x * factor + bias

    # Prepare inputs
    x_orig = torch.randn(3, 3, device='xla', requires_grad=True)
    bias_orig = torch.randn(3, 3, device='xla', requires_grad=True)
    factor_val = 2.5  # Non-tensor kwarg

    x_pure = x_orig.clone().detach().requires_grad_(True)
    bias_pure = bias_orig.clone().detach().requires_grad_(True)

    # Act
    # Forward pass
    output_orig = fn(x_orig, factor=factor_val, bias=bias_orig)
    output_pure = assume_pure(fn)(x_pure, factor=factor_val, bias=bias_pure)
    torch.testing.assert_close(
        output_pure,
        output_orig,
        msg="Forward outputs do not match",
        check_device=False)

    # Backward pass
    output_orig.sum().backward()
    output_pure.sum().backward()
    torch_xla.sync()

    # Assert
    # Check gradients
    assert_gradients_close(self, x_pure, x_orig)
    assert_gradients_close(self, bias_pure, bias_orig)
    # Factor is not a tensor, so it won't have a .grad attribute. Nothing to check here.

  def test_assume_pure_no_grads_needed(self):
    """Tests a function where no inputs require gradients."""

    # Arrange
    def original_func(a, b):
      return torch.cos(a) + torch.sin(b)

    # Prepare inputs
    a_orig = torch.randn(3, 3, device='xla', requires_grad=False)
    b_orig = torch.randn(3, 3, device='xla', requires_grad=False)
    a_pure = a_orig.clone().detach().requires_grad_(False)
    b_pure = b_orig.clone().detach().requires_grad_(False)

    # Act
    # Forward pass
    output_orig = original_func(a_orig, b_orig)
    output_pure = assume_pure(original_func)(a_pure, b_pure)
    torch_xla.sync()

    # Assert
    # Check outputs
    torch.testing.assert_close(
        output_pure,
        output_orig,
        msg="Forward outputs do not match",
        check_device=False)

    # Check gradients
    self.assertFalse(output_orig.requires_grad)
    self.assertFalse(output_pure.requires_grad)
    self.assertIsNone(a_orig.grad)
    self.assertIsNone(b_orig.grad)
    self.assertIsNone(a_pure.grad)
    self.assertIsNone(b_pure.grad)

  def test_composibility_with_call_jax(self):

    def jax_func(a, b):
      return jnp.dot(a, b)

    def f(a, b):
      return xb.call_jax(jax_func, (a, b))

    a = torch.randn(3, 3, device=self.device)
    b = torch.randn(3, 3, device=self.device)

    output_pure = assume_pure(f)(a, b)
    torch.testing.assert_close(
        output_pure,
        a @ b,
        msg="Forward outputs do not match",
        check_device=False)

  def test_assume_pure_recursive(self):

    @assume_pure
    def torch_func(a, b):
      return torch.matmul(a, b)

    def f(a, b):
      y = torch_func(a, b)
      return y + 1

    a = torch.randn(3, 3, device=self.device)
    b = torch.randn(3, 3, device=self.device)

    output_pure = assume_pure(f)(a, b)
    torch.testing.assert_close(
        output_pure,
        a @ b + 1,
        msg="Forward outputs do not match",
        check_device=False)


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    name='benchmark_iterations',
    default=3,
    help='Number of iterations to run the tracing benchmark test.')


class TracingBenchmark(absltest.TestCase):

  def test_trace_transformer_with_spda_attention(self):
    num_iterations = FLAGS.benchmark_iterations
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
    @assume_pure
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
  torch_xla._XLAC._xla_set_mat_mul_precision('highest')
  jax_import_guard()
  import torchax
  import jax.numpy as jnp
  torchax.enable_accuracy_mode()
  absltest.main()
