import unittest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.test.test_utils as test_utils
from torch_xla.experimental.gradient_accumulation import gradient_accumulation

from test_utils import XlaTestCase  # type:ignore


class SimpleModel(torch.nn.Module):

  def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
    super(SimpleModel, self).__init__()
    self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
    self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    return self.fc2(x)


class GradAccumulationTest(XlaTestCase):

  def setUp(self):
    self.device = xm.xla_device()
    torch.manual_seed(123)

  def test_basic(self):
    """Compare results with and without the XLA loop"""
    batch_size = 8
    hidden_dim = 20
    input_dim = 10
    output_dim = 5

    inputs = torch.randn(batch_size, input_dim).to(self.device)
    targets = torch.randn(batch_size, output_dim).to(self.device)

    def train_step_fw(input_batch, target_batch, carried_tensor):
      output = model_ga(input_batch)
      loss = torch.nn.functional.mse_loss(output, target_batch)
      new_carried_tensor = carried_tensor + 5
      return loss, new_carried_tensor

    # Gradient accumulation with XLA loop
    torch.manual_seed(43)
    model_ga = SimpleModel(input_dim, hidden_dim, output_dim).to(self.device)
    carried_tensor_ga = torch.tensor([5, 5]).to(self.device)

    accumulated_loss_ga, accum_carried_tensor_ga = gradient_accumulation(
        train_step_fw, (inputs, targets), model_ga, carried_tensor_ga)

    torch_xla.sync()

    # Traditional accumulation
    torch.manual_seed(43)
    model_manual = SimpleModel(input_dim, hidden_dim,
                               output_dim).to(self.device)
    carried_tensor_manual = torch.tensor([5, 5]).to(self.device)

    accumulated_loss_manual = torch.tensor(0.0).to(self.device)
    for i in range(batch_size):
      loss, carried_tensor_manual = train_step_fw(inputs[i:i + 1],
                                                  targets[i:i + 1],
                                                  carried_tensor_manual)
      loss = loss / batch_size
      loss.backward()
      accumulated_loss_manual += loss.detach()

    torch_xla.sync()

    # Compare losses, carried tensors and resulting gradients
    super().compareResults([accumulated_loss_ga], [accumulated_loss_manual])
    super().compareResults([accum_carried_tensor_ga], [carried_tensor_manual])
    super().compareResults(model_ga.parameters(), model_manual.parameters())

  def test_with_carried_tensors(self):
    """Test gradient accumulation with carried tensors, including with RNG"""
    batch_size = 2
    hidden_dim = 20
    input_dim = 10
    output_dim = 5

    model = SimpleModel(input_dim, hidden_dim, output_dim).to(self.device)

    inputs = torch.randn(batch_size, input_dim).to(self.device)
    targets = torch.randn(batch_size, output_dim).to(self.device)

    # Carried tensors
    counter = torch.tensor(0).to(self.device)
    tensor0 = torch.tensor(0.0).to(self.device)
    tensor0_baseline = tensor0.clone()

    # Define train step function that updates the carried tensor. In the case of
    # RNG, we negate the previous value, in order to validate that we get unique
    # RNG seeds for each iteration.
    def train_step_fw(input_batch, target_batch, counter, tensor0):
      output = model(input_batch)
      loss = torch.nn.functional.mse_loss(output, target_batch)
      # Update counter
      new_counter = counter + 1
      new_tensor0 = torch.rand_like(tensor0, device=self.device) - tensor0
      return loss, new_counter, new_tensor0

    # Run gradient accumulation
    accumulated_loss, final_counter, final_tensor0 = gradient_accumulation(
        train_step_fw, (inputs, targets), model, counter, tensor0)

    torch_xla.sync()

    self.assertEqual(final_counter.item(), batch_size)
    # Ensure that the result is not 0, showcasing that the RNG is unique
    # per iteration.
    self.assertNotEqual(final_tensor0.item(), 0.0)

  def test_error_empty_iterable_tensors(self):
    """Test that empty iterable_tensors raises an error."""
    model = SimpleModel().to(self.device)

    def train_step_fw():
      pass

    with self.assertRaises(ValueError):
      gradient_accumulation(train_step_fw, [], model)

  def test_error_mutated_input_tensors(self):
    """Test that mutating input tensors raises an error."""
    batch_size = 2
    hidden_dim = 20
    input_dim = 10
    output_dim = 5

    model = SimpleModel(input_dim, hidden_dim, output_dim).to(self.device)

    inputs = torch.randn(batch_size, input_dim).to(self.device)
    targets = torch.randn(batch_size, output_dim).to(self.device)
    counter = torch.tensor(0).to(self.device)

    def train_step_fw(input_batch, target_batch, counter):
      output = model(input_batch)
      loss = torch.nn.functional.mse_loss(output, target_batch)
      # In-place mutation of an input tensor.
      counter += 1
      return loss, counter

    with self.assertRaises(AssertionError):
      accumulated_loss, final_counter = gradient_accumulation(
          train_step_fw, (inputs, targets), model, counter)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
