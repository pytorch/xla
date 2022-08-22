from absl.testing import absltest, parameterized
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
from torch_xla.experimental import pjrt

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def _init_xla_backend(init_file: str):
  # TODO: fix these
  rank = xm.get_ordinal()
  world_size = xm.xrt_world_size()

  dist.init_process_group(
    "xla",
    init_method=f"file://{init_file}",
    rank=rank,
    world_size=world_size)

class TestPjRtDistributedDataParallel(parameterized.TestCase):

  @staticmethod
  def _ddp_step(init_file: str):
    _init_xla_backend(init_file)

    device = xm.xla_device()
    model = ToyModel().to(device)
    ddp_model = DDP(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10, device=device))

    labels = torch.randn(20, 5).to(device)
    loss_fn(outputs, labels).backward()
    optimizer.step()

  def test_ddp_step(self):
    pjrt.run_multiprocess(
      self._ddp_step,
      self.create_tempfile().full_path)

  @staticmethod
  def _ddp_loss(init_file: str, barrier: bool):
    _init_xla_backend(init_file)

    device = xm.xla_device()

    model = nn.Linear(10, 10).to(device)
    model = DDP(model)

    optimizer = optim.SGD(model.parameters(), lr=1e-100)
    loss_fn = nn.MSELoss()

    for step in range(100):
      x = torch.ones(20, 10, device=device)
      labels = torch.zeros(20, 10, device=device)
      if barrier:
        xm.mark_step()

      optimizer.zero_grad()
      outputs = model(x)
      loss = loss_fn(outputs, labels)
      loss.backward()

      optimizer.step()

      if not loss.isfinite():
        raise ValueError('infinite loss ({}) on step {}'.format(loss, step))

  @parameterized.named_parameters(("barrier", True), ("nobarrier", False))
  def test_ddp_loss(self, barrier):
    pjrt.run_multiprocess(
      self._ddp_loss,
      self.create_tempfile().full_path,
      barrier)

  @staticmethod
  def _ddp_loss_nosync(init_file: str):
    _init_xla_backend(init_file)

    device = xm.xla_device()

    model = nn.Linear(10, 10).to(device)
    model = DDP(model)

    optimizer = optim.SGD(model.parameters(), lr=1e-100)
    loss_fn = nn.MSELoss()

    for step in range(100):
      x = torch.ones(20, 10, device=device)
      labels = torch.zeros(20, 10, device=device)
      xm.mark_step()

      with model.no_sync():
        outputs = model(x)
        loss = loss_fn(outputs, labels)
        loss.backward()

      xm.optimizer_step(optimizer, pin_layout=False)

      if not loss.isfinite():
        raise ValueError('infinite loss ({}) on step {}'.format(loss, step))

  def test_ddp_loss_nosync(self):
    pjrt.run_multiprocess(
      self._ddp_loss_nosync,
      self.create_tempfile().full_path)


if __name__ == "__main__":
  absltest.main()
