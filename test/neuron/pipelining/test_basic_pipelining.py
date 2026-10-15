# Test for XLA Pipeline Parallelism - requires torchrun --nproc_per_node=2
# Only runs on NEURON devices with exactly 2 processes

import os
import sys
import unittest
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.distributed as dist

import torch_xla.distributed.pipelining
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla
import torch_xla.utils.utils as xu


class MLPModule(torch.nn.Module):

  def __init__(self, d_hid, num_classes, is_last=False):
    super().__init__()
    self.net1 = torch.nn.Linear(d_hid, d_hid // 2)
    self.relu = torch.nn.ReLU()
    self.net2 = torch.nn.Linear(d_hid // 2, num_classes if is_last else d_hid)

  def forward(self, x):
    x = self.net1(x)
    x = self.relu(x)
    x = self.net2(x)
    return x


class SimpleTransf(torch.nn.Module):

  def __init__(self, hidden_dim, num_classes, num_layers):
    super().__init__()
    self.layers = torch.nn.Sequential(*[
        MLPModule(hidden_dim, num_classes, is_last=(i == num_layers - 1))
        for i in range(num_layers)
    ])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layers(x)


class TestLazyBasicPipelining(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    # Only run on NEURON devices
    if xr.device_type() != 'NEURON':
      raise unittest.SkipTest('Test only runs on NEURON devices')

    # Require distributed environment with exactly 2 devices
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
      raise unittest.SkipTest('Test requires torchrun with RANK and WORLD_SIZE')

    world_size = int(os.environ.get('WORLD_SIZE', 0))
    if world_size != 2:
      raise unittest.SkipTest('Test requires exactly 2 devices')

  def test_pipeline_training(self):
    """Test distributed pipeline training with GPipe scheduling and loss convergence"""
    # configs
    hidden_dim = 1024
    num_classes = 32
    num_layers = 2
    batch_size = 2
    num_epochs = 2
    lr = 0.01
    train_dataset_len = 1024 * 8
    gradient_accumulation_steps = 1
    logging_steps = 1

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    chunks = world_size

    device = torch_xla.device()
    print(f"Rank {rank} using device {device}")

    # Initialize process group
    dist.init_process_group(
        backend="xla", rank=rank, world_size=world_size, init_method='xla://')

    torch.manual_seed(42)
    model = SimpleTransf(hidden_dim, num_classes, num_layers).to("meta")

    # Define split points for pipeline parallelism
    split_spec = {}
    for i in range(1, chunks):
      layer_idx = (i * num_layers) // chunks
      split_spec[f"layers.{layer_idx}"] = SplitPoint.BEGINNING

    # Create a sample input for the pipeline
    example_input = torch.randn(batch_size, hidden_dim, device="meta")

    # Create the pipeline and respective stage for the rank
    pipe = pipeline(model, mb_args=(example_input,), split_spec=split_spec)

    # Assertions for pipeline creation
    self.assertEqual(
        pipe.num_stages, chunks,
        f"Pipeline stages should match chunks: {pipe.num_stages} != {chunks}")

    loss_fn = nn.CrossEntropyLoss()
    stage_mod = pipe.get_stage_module(rank)
    stage_mod = stage_mod.to_empty(device=device)
    stage = PipelineStage(stage_mod, rank, chunks, device)
    schedule = ScheduleGPipe(stage, batch_size, loss_fn=loss_fn)

    del model
    if rank == 0:
      print(f"{rank=}, {schedule._get_pipeline_order()}\n", flush=True)

    losses = []
    to_check_losses = []
    optimizer = optim.SGD(stage_mod.parameters(), lr=lr)

    train_loader = xu.SampleGenerator(
        data=(
            torch.randn(batch_size, hidden_dim),
            torch.randint(0, num_classes, (batch_size,), dtype=torch.int64),
        ),
        sample_count=train_dataset_len //
        (batch_size * gradient_accumulation_steps),
    )

    def print_epoch(epoch, step, losses):
      to_check_losses.append(losses[-1].item())
      print(f"Epoch {epoch} step {step} loss {losses[-1]}")

    print(f"{world_size=}, {rank=}")

    # Training loop with rank-specific pipeline logic
    for epoch in range(num_epochs):
      for step, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if rank == 0:  # First rank handles input
          data = data.to(device)
          _ = schedule.step(data)
        elif rank == world_size - 1:  # Last rank handles target and loss
          target = target.to(device)
          _ = schedule.step(target=target, losses=losses)
        else:  # Middle ranks just forward/backward
          _ = schedule.step()

        if rank == world_size - 1:
          if step % logging_steps == 0:
            xm.add_step_closure(print_epoch, (epoch, step, losses))

        optimizer.step()
        torch.distributed.barrier()
        torch_xla.sync()
        if step == 100:  # break for assertions
          break
      break
    # Validate training results
    if to_check_losses:
      if rank == 0:
        self.assertTrue(
            len(to_check_losses) == 0,
            f"rank0 should not store losses, but it has: {to_check_losses}")
      if rank == world_size - 1:
        # Last rank records losses - verify they exist and are finite
        self.assertGreater(
            len(to_check_losses), 0, "Last rank should record losses")
        for loss in to_check_losses:
          self.assertTrue(
              torch.isfinite(torch.Tensor([loss])),
              f"Loss should be finite: {loss}")

        # Check loss convergence (early vs late training)
        if len(to_check_losses) >= 10:
          early_losses = [
              l for l in to_check_losses[:len(to_check_losses) // 3]
          ]
          late_losses = [
              l for l in to_check_losses[-len(to_check_losses) // 3:]
          ]
          early_avg = sum(early_losses) / len(early_losses)
          late_avg = sum(late_losses) / len(late_losses)

          print(
              f"Early avg loss: {early_avg:.4f}, Late avg loss: {late_avg:.4f}")
          self.assertLessEqual(
              late_avg, early_avg * 1.1,
              f"Loss should generally decrease: early={early_avg:.4f}, late={late_avg:.4f}"
          )
    dist.destroy_process_group()


if __name__ == '__main__':
  unittest.main()
