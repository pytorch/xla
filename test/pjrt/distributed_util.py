from absl import logging
import copy
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
from torch_xla.experimental import pjrt


def init_xla_backend(init_file: str):
  rank = xm.get_ordinal()
  world_size = xm.xrt_world_size()

  dist.init_process_group(
      "xla",
      init_method=f"file://{init_file}" if init_file is not None else None,
      rank=rank,
      world_size=world_size)
  return rank, world_size


def assert_all_close(parameters_a, parameters_b):
  for param_a, param_b in zip(parameters_a, parameters_b):
    assert torch.allclose(param_a.cpu(), param_b.cpu())


def train_step(model, inputs, labels, optimizer, loss_fn):
  optimizer.zero_grad()

  outputs = model(inputs)
  loss = loss_fn(outputs, labels)
  loss.backward()
  optimizer.step()

  xm.mark_step()

  return loss

def ddp_correctness(init_file: str):
  rank, world_size = init_xla_backend(init_file)

  device = xm.xla_device()

  cpu_model = nn.Linear(10, 10)
  # TODO(@alanwaketan): Investigate whether we can omit the gradient_as_bucket_view option.
  ddp_model = DDP(
      copy.deepcopy(cpu_model).to(device), gradient_as_bucket_view=True)

  cpu_optimizer = optim.SGD(cpu_model.parameters(), lr=1e-100)
  ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=1e-100)
  loss_fn = nn.MSELoss()

  local_batch_size = 2
  global_batch_size = local_batch_size * world_size
  offset = rank * local_batch_size
  # Lower range probably makes sense too. Anyway, stick to 100 as the original PoC.
  for step in range(100):
    # To make torch.randn produce same results across devices.
    torch.manual_seed(2022 + step)

    cpu_inputs = torch.randn(global_batch_size, 10)
    cpu_labels = torch.randn(global_batch_size, 10)
    cpu_loss = train_step(cpu_model, cpu_inputs, cpu_labels, cpu_optimizer,
                            loss_fn)

    ddp_inputs = copy.deepcopy(cpu_inputs[offset:offset +
                                          local_batch_size]).to(device)
    ddp_labels = copy.deepcopy(cpu_labels[offset:offset +
                                          local_batch_size]).to(device)
    ddp_loss = train_step(ddp_model, ddp_inputs, ddp_labels, ddp_optimizer,
                            loss_fn)
    with torch.no_grad():
      ddp_loss = ddp_loss / world_size
      dist.all_reduce(ddp_loss)

    # TODO(@alanwaketan): Investigate why the atol here is this low.
    assert torch.allclose(cpu_loss, ddp_loss, atol=1e-02)
    assert_all_close(cpu_model.parameters(), ddp_model.parameters())
    # To display the below messages, set '--verbosity=1'.
    logging.debug(
        "iteration %d: cpu_loss = %f, ddp_loss = %f, cpu_model.parameters() ~= ddp_model.parameters()",
        step, cpu_loss, ddp_loss)
