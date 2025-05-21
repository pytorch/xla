import argparse
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--verbosity', type=int, default=2)
FLAGS, leftovers = parser.parse_known_args()
sys.argv = [sys.argv[0]] + leftovers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import unittest
import numpy as np
try:
  from torch_xla.amp import syncfree
except ImportError:
  assert False, "Missing package syncfree; the package is available in torch-xla>=1.11"


class MNIST(nn.Module):

  def __init__(self):
    super(MNIST, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.bn1 = nn.BatchNorm2d(10)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.bn2 = nn.BatchNorm2d(20)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = self.bn1(x)
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = self.bn2(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


class TestSyncFreeOptimizerBase(unittest.TestCase):

  def setUp(self):
    super(TestSyncFreeOptimizerBase, self).setUp()

  def tearDown(self):
    super(TestSyncFreeOptimizerBase, self).tearDown()

  def _test_optimizer(self,
                      syncfree_optim_cls,
                      ref_optim_cls,
                      optim_kwargs={'lr': 1e-2}):
    device = xm.xla_device()
    loss_fn = nn.NLLLoss()
    # syncfree model
    torch.manual_seed(0)
    syncfree_model = MNIST().train().to(device)
    syncfree_optimizer = syncfree_optim_cls(syncfree_model.parameters(),
                                            **optim_kwargs)
    # reference model
    torch.manual_seed(0)
    ref_model = MNIST().train().to(device)
    ref_optimizer = ref_optim_cls(ref_model.parameters(), **optim_kwargs)
    # fake data
    data = torch.rand(32, 1, 28, 28).to(device)
    target = torch.zeros(32).to(device)
    # training loop
    for i in range(10):
      # syncfree step
      syncfree_optimizer.zero_grad()
      syncfree_output = syncfree_model(data)
      syncfree_loss = loss_fn(syncfree_output, target)
      syncfree_loss.backward()
      # mimick nan in the gradients
      if i % 2 == 0:
        xm._fetch_gradients(syncfree_optimizer)[0].mul_(torch.nan)
        found_inf = torch.tensor(1.0).to(device)
      else:
        found_inf = torch.tensor(0.0).to(device)
      xm.optimizer_step(
          syncfree_optimizer, optimizer_args={"found_inf": found_inf})
      torch_xla.sync()
      # reference step
      ref_optimizer.zero_grad()
      ref_output = ref_model(data)
      ref_loss = loss_fn(ref_output, target)
      ref_loss.backward()
      # mimick the effect of found_inf tensor
      if i % 2 != 0:
        xm.optimizer_step(ref_optimizer)
      torch_xla.sync()
      # check loss
      np.testing.assert_allclose(
          ref_loss.cpu().detach().numpy(),
          syncfree_loss.cpu().detach().numpy(),
          rtol=1e-2,
          atol=1e-1)

    # check weight
    for p, p_ref in zip(syncfree_model.parameters(), ref_model.parameters()):
      np.testing.assert_allclose(
          p.cpu().detach().numpy(),
          p_ref.cpu().detach().numpy(),
          rtol=1e-2,
          atol=1e-1)


class TestSyncFreeSGD(TestSyncFreeOptimizerBase):

  def test_optimizer(self):
    self._test_optimizer(syncfree.SGD, torch.optim.SGD, {
        "lr": 1e-2,
        "momentum": 0.5,
    })
    self._test_optimizer(syncfree.SGD, torch.optim.SGD, {
        "lr": 1e-2,
        "weight_decay": 0.1,
    })
    self._test_optimizer(syncfree.SGD, torch.optim.SGD, {
        "lr": 1e-2,
        "momentum": 0.5,
        "weight_decay": 0.1,
        "dampening": 0.1,
    })
    self._test_optimizer(syncfree.SGD, torch.optim.SGD, {
        "lr": 1e-2,
        "momentum": 0.5,
        "weight_decay": 0.1,
        "nesterov": True,
    })
    self._test_optimizer(
        syncfree.SGD, torch.optim.SGD, {
            "lr": 1e-2,
            "momentum": 0.5,
            "weight_decay": 0.1,
            "nesterov": True,
            "maximize": True,
        })


class TestSyncFreeAdam(TestSyncFreeOptimizerBase):

  def _test_adam_optimizer_helper(self, optim, optim_ref):
    self._test_optimizer(optim, optim_ref, {
        "lr": 1e-3,
        "betas": (0.9, 0.99),
    })
    self._test_optimizer(optim, optim_ref, {
        "lr": 1e-3,
        "betas": (0.7, 0.77),
        "weight_decay": 1e-4,
    })
    self._test_optimizer(optim, optim_ref, {
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-4,
    })
    self._test_optimizer(optim, optim_ref, {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "weight_decay": 0.1,
    })
    self._test_optimizer(optim, optim_ref, {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "weight_decay": 0.1,
        "amsgrad": True,
    })
    self._test_optimizer(optim, optim_ref, {
        "lr": 1e-4,
        "betas": (0.7, 0.799),
        "weight_decay": 0.01,
        "amsgrad": True,
    })
    self._test_optimizer(
        optim, optim_ref, {
            "lr": 1e-4,
            "betas": (0.7, 0.799),
            "weight_decay": 0.01,
            "amsgrad": True,
            "maximize": True,
        })

  def test_adam_optimizer(self):
    self._test_adam_optimizer_helper(syncfree.Adam, torch.optim.Adam)
    self._test_adam_optimizer_helper(syncfree.AdamW, torch.optim.AdamW)


if __name__ == "__main__":
  test = unittest.main(verbosity=FLAGS.verbosity, exit=False)
  sys.exit(0 if test.result.wasSuccessful() else 1)
