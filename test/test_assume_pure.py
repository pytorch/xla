from absl.testing import absltest

import torch
import torch.nn as nn
import torch_xla
from torch_xla.experimental.assume_pure import assume_pure
import torch_xla.core.xla_builder as xb
from torch_xla._internal.jax_workarounds import jax_import_guard


# Helper function to compare gradients, handling None cases
def assert_gradients_close(test_case, tensor1, tensor2):
  grad1 = tensor1.grad
  grad2 = tensor2.grad
  if grad1 is None and grad2 is None:
    return  # Both are None, which is expected if requires_grad=False or disconnected
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


class TestJaxInterop(absltest.TestCase):

  def setUp(self):
    # Ensure we're using the XLA device for tests
    self.device = torch_xla.device()

  def test_assume_pure_basic(self):

    @assume_pure
    def simple_torch_function(a, b):
      return torch.sin(a @ b)

    a = torch.ones((3, 3), device='xla', requires_grad=True)
    o = simple_torch_function(a, a)
    o.sum().backward()

    torch_xla.sync()
    torch.testing.assert_close(
        o, torch.sin(torch.ones(3, 3) @ torch.ones(3, 3)), check_device=False)

  def test_assume_pure_module(self):
    model = nn.Linear(3, 3).to('xla')

    @assume_pure
    def simple_torch_function(params, x):
      return torch.func.functional_call(model, params, x)

    a = torch.ones((3, 3), device='xla', requires_grad=True)
    o = simple_torch_function(dict(model.named_parameters()), a)
    o.sum().backward()

    torch_xla.sync()

    torch.testing.assert_close(
        o, model(torch.ones(3, 3).to('xla')), check_device=False)

  def test_assume_pure_avoid_retracing_avoid_rejit(self):
    starting_lowerings = xb._jax_to_hlo_cache_num_misses()
    trace_counter = 0

    @assume_pure
    def simple_torch_function(a, b):
      nonlocal trace_counter
      trace_counter += 1
      return torch.sin(a @ b)

    # Simulate a training loop.
    for _ in range(5):
      a = torch.ones((3, 3), device='xla', requires_grad=True)
      o = simple_torch_function(a, a)
      o.sum().backward()
      torch_xla.sync()

    ending_lowerings = xb._jax_to_hlo_cache_num_misses()

    # Check that we only trace once.
    self.assertEqual(trace_counter, 1)

    # Check that we only lower to HLO twice (once for forward, once for backward).
    self.assertEqual(ending_lowerings - starting_lowerings, 2)

  def test_assume_pure_matmul_grads(self):
    """Tests matmul with all inputs requiring gradients."""

    # Define original and decorated functions
    def original_matmul(a, b):
      return a @ b

    @assume_pure
    def decorated_matmul(a, b):
      # Note: The function wrapped by assume_pure should ideally use torch ops
      # that have XLA lowering support for efficiency, which matmul does.
      return a @ b

    # Prepare inputs (cloned for independent grad computation)
    a_orig = torch.randn(4, 5, device=self.device, requires_grad=True)
    b_orig = torch.randn(5, 3, device=self.device, requires_grad=True)
    a_dec = a_orig.clone().detach().requires_grad_(True)
    b_dec = b_orig.clone().detach().requires_grad_(True)

    # --- Forward Pass ---
    output_orig = original_matmul(a_orig, b_orig)
    output_dec = decorated_matmul(a_dec, b_dec)

    # Check forward pass equivalence
    torch.testing.assert_close(
        output_orig,
        output_dec,
        msg="Forward outputs do not match",
        check_device=False)

    # --- Backward Pass ---
    loss_orig = output_orig.sum()
    loss_dec = output_dec.sum()

    loss_orig.backward()
    loss_dec.backward()
    torch_xla.sync()  # Use mark_step or sync to ensure computations complete

    # Check gradients
    assert_gradients_close(self, a_orig, a_dec)
    assert_gradients_close(self, b_orig, b_dec)

  def test_assume_pure_einsum_grads(self):
    """Tests einsum with all inputs requiring gradients."""

    def original_einsum(x, y):
      return torch.einsum('bij,bjk->bik', x, y)

    @assume_pure
    def decorated_einsum(x, y):
      return torch.einsum('bij,bjk->bik', x, y)

    # Prepare inputs
    x_orig = torch.randn(2, 3, 4, device=self.device, requires_grad=True)
    y_orig = torch.randn(2, 4, 5, device=self.device, requires_grad=True)
    x_dec = x_orig.clone().detach().requires_grad_(True)
    y_dec = y_orig.clone().detach().requires_grad_(True)

    # --- Forward Pass ---
    output_orig = original_einsum(x_orig, y_orig)
    output_dec = decorated_einsum(x_dec, y_dec)
    torch.testing.assert_close(
        output_orig,
        output_dec,
        msg=lambda msg: f"Forward outputs do not match: {msg}",
        check_device=False)

    # --- Backward Pass ---
    output_orig.sum().backward()
    output_dec.sum().backward()
    torch_xla.sync()

    # Check gradients
    assert_gradients_close(self, x_orig, x_dec)
    assert_gradients_close(self, y_orig, y_dec)

  def test_assume_pure_partial_grads_args(self):
    """Tests a function where only some positional inputs require gradients."""

    def original_func(a, b, c):  # a, c require grad; b does not
      return a * torch.tanh(b) + c**2

    @assume_pure
    def decorated_func(a, b, c):
      return a * torch.tanh(b) + c**2

    # Prepare inputs
    torch_xla.manual_seed(42)
    a_orig = torch.randn(
        3, 3, device=self.device, requires_grad=True, dtype=torch.bfloat16)
    b_orig = torch.randn(
        3, 3, device=self.device, requires_grad=False,
        dtype=torch.bfloat16)  # No grad for b
    c_orig = torch.randn(
        3, 3, device=self.device, requires_grad=True, dtype=torch.bfloat16)

    a_dec = a_orig.clone().detach().requires_grad_(True)
    b_dec = b_orig.clone().detach().requires_grad_(False)  # Match requires_grad
    c_dec = c_orig.clone().detach().requires_grad_(True)

    # --- Forward Pass ---
    output_orig = original_func(a_orig, b_orig, c_orig)
    output_dec = decorated_func(a_dec, b_dec, c_dec)
    torch.testing.assert_close(
        output_orig,
        output_dec,
        msg="Forward outputs do not match",
        check_device=False)

    # --- Backward Pass ---
    output_orig.sum().backward()
    output_dec.sum().backward()
    torch_xla.sync()

    # Check gradients
    assert_gradients_close(self, a_orig, a_dec)
    assert_gradients_close(self, b_orig, b_dec)  # Should both be None
    assert_gradients_close(self, c_orig, c_dec)

    self.assertIsNone(b_orig.grad, "b_orig should not have grad")
    self.assertIsNone(b_dec.grad, "b_dec should not have grad")

  def test_assume_pure_partial_grads_kwargs(self):
    """Tests a function where inputs requiring gradients are passed via kwargs."""

    def original_func(x, *, factor,
                      bias):  # x, bias require grad; factor does not
      # factor is a non-tensor kwarg, bias is a tensor kwarg
      return x * factor + bias

    @assume_pure
    def decorated_func(x, *, factor, bias):
      return x * factor + bias

    # Prepare inputs
    x_orig = torch.randn(3, 3, device=self.device, requires_grad=True)
    bias_orig = torch.randn(3, 3, device=self.device, requires_grad=True)
    factor_val = 2.5  # Non-tensor kwarg

    x_dec = x_orig.clone().detach().requires_grad_(True)
    bias_dec = bias_orig.clone().detach().requires_grad_(True)

    # --- Forward Pass ---
    output_orig = original_func(x_orig, factor=factor_val, bias=bias_orig)
    output_dec = decorated_func(x_dec, factor=factor_val, bias=bias_dec)
    torch.testing.assert_close(
        output_orig,
        output_dec,
        msg="Forward outputs do not match",
        check_device=False)

    # --- Backward Pass ---
    output_orig.sum().backward()
    output_dec.sum().backward()
    torch_xla.sync()

    # Check gradients
    assert_gradients_close(self, x_orig, x_dec)
    assert_gradients_close(self, bias_orig, bias_dec)
    # Factor is not a tensor, so it won't have a .grad attribute

  def test_assume_pure_no_grads_needed(self):
    """Tests a function where no inputs require gradients."""

    def original_func(a, b):
      return torch.cos(a) + torch.sin(b)

    @assume_pure
    def decorated_func(a, b):
      return torch.cos(a) + torch.sin(b)

    # Prepare inputs
    a_orig = torch.randn(3, 3, device=self.device, requires_grad=False)
    b_orig = torch.randn(3, 3, device=self.device, requires_grad=False)
    a_dec = a_orig.clone().detach().requires_grad_(False)
    b_dec = b_orig.clone().detach().requires_grad_(False)

    # --- Forward Pass ---
    output_orig = original_func(a_orig, b_orig)
    output_dec = decorated_func(a_dec, b_dec)
    torch.testing.assert_close(
        output_orig,
        output_dec,
        msg="Forward outputs do not match",
        check_device=False)

    # --- Backward Pass (Optional Check) ---
    # Cannot call backward if output doesn't require grad
    self.assertFalse(output_orig.requires_grad)
    self.assertFalse(output_dec.requires_grad)

    # Explicitly check grads are None
    self.assertIsNone(a_orig.grad)
    self.assertIsNone(b_orig.grad)
    self.assertIsNone(a_dec.grad)
    self.assertIsNone(b_dec.grad)


if __name__ == "__main__":
  torch.set_default_dtype(torch.bfloat16)
  torch.manual_seed(42)
  torch_xla.manual_seed(42)
  torch_xla._XLAC._xla_set_mat_mul_precision('highest')
  jax_import_guard()
  import torchax
  torchax.enable_accuracy_mode()
  absltest.main()
