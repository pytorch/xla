import unittest
import torch
import torchax as tx
import torchax.export
import torchax.train
from torch.testing._internal.common_utils import TestCase


class TrainTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(0)
    torchax.enable_accuracy_mode()

  def test_scan_module(self):
    x = torch.arange(300).reshape(3, 100).to(torch.float32)
    layers = [
        torch.nn.Linear(100, 100),
        torch.nn.Linear(100, 100),
        torch.nn.Linear(100, 100),
        torch.nn.Linear(100, 100),
    ]
    # repetitively applies the linear
    result = x
    for layer in layers:
      result = layer(result)

    model = tx.train.ScannedModule(layers)

    with torchax.default_env():
      x = x.to('jax')
      model.to('jax')
      result2 = model(x)
      torch.testing.assert_allclose(result, result2.to('cpu'))

  def test_train_step_can_run(self):
    import optax
    with torchax.default_env():
      model = torch.nn.Linear(100, 100)
      model.to('jax')
      weights = model.state_dict()
      x = torch.randn(2, 100).to('jax')
      y = torch.tensor([1, 2]).to('jax')

      def model_fn(weight, buffers, args):
        return torch.func.functional_call(model, weight, args)

      loss_fn = torch.nn.CrossEntropyLoss()

      optimizer = optax.adam(0.01)
      opt_state = tx.interop.call_jax(optimizer.init, weights)

      step = tx.train.make_train_step(model_fn, loss_fn, optimizer)
      print(step(weights, {}, opt_state, x, y))


if __name__ == '__main__':
  unittest.main()
