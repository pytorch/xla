import copy
import threading
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.xla_backend


# The followings are helpers useful for debugging purpose.
def comp_hook(state: object,
              bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
  """
  Debug utils. Please refer to DistributedDataParallel.register_comm_hook to learn
  how to use it.
  """
  print("comp_hook called.")
  fut = torch.futures.Future()
  fut.set_result(bucket.buffer())
  return fut


def calculate_model_size(model):
  """
  Debug utils. Calculate the given model's size in mb.
  """
  param_size = 0
  for param in model.parameters():
    param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))


class LargeNet(nn.Module):

  def __init__(self):
    super(LargeNet, self).__init__()
    self.net1 = nn.Linear(10, 1000)
    self.net2 = nn.Linear(1000, 1000)
    self.net3 = nn.Linear(1000, 1000)
    self.relu = nn.ReLU()
    self.net4 = nn.Linear(1000, 10)

  def forward(self, x):
    output1 = self.relu(self.net1(x))
    output2 = self.relu(self.net2(output1))
    output3 = self.relu(self.net3(output2))
    return self.net4(output3)


class SmallNet(nn.Module):

  def __init__(self):
    super(SmallNet, self).__init__()
    self.net = nn.Linear(10, 10)

  def forward(self, x):
    return self.net(x)


def assert_all_close(parameters_a, parameters_b):
  for param_a, param_b in zip(parameters_a, parameters_b):
    assert torch.allclose(param_a.cpu(), param_b.cpu(), atol=1e-3)


def train_step(model, inputs, labels, optimizer, loss_fn):
  optimizer.zero_grad()

  outputs = model(inputs)
  loss = loss_fn(outputs, labels)
  loss.backward()
  optimizer.step()

  xm.mark_step()

  return loss


init_lock = threading.Lock()


def ddp_correctness(init_method: str = 'env://',
                    use_large_net: bool = False,
                    debug: bool = False):
  if init_method == 'env://':
    rank = xr.global_ordinal()
    world_size = xr.world_size()
    dist.init_process_group(
        "xla", init_method=init_method, rank=rank, world_size=world_size)
  else:
    dist.init_process_group("xla", init_method=init_method)

  rank, world_size = dist.get_rank(), dist.get_world_size()
  device = xm.xla_device()

  # Module initialization is not thread safe. Force threads to initialize one
  # at a time with the same seed
  with init_lock:
    torch.manual_seed(2022)
    steps = 100
    cpu_model = SmallNet()
    if use_large_net:
      steps = 5  # To save test time.
      cpu_model = LargeNet()

  # TODO: There're issues in the captured graph when gradient_as_bucket_view is True
  # bucket_cap_mb is set to 1 mb such that we can still have multiple all_reduces while avoiding
  # using models that are too larger (25 mb).
  # To be noted, DDP currently uses one bucket for the first iteration. See pytorch#73732.
  ddp_model = DDP(copy.deepcopy(cpu_model).to(device), bucket_cap_mb=1)
  # ddp_model.register_comm_hook(state=None, hook=comp_hook)

  cpu_optimizer = optim.SGD(cpu_model.parameters(), lr=1e-1)
  ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=1e-1)
  loss_fn = nn.MSELoss()

  local_batch_size = 2
  global_batch_size = local_batch_size * world_size
  offset = rank * local_batch_size
  for step in range(steps):
    # Use a local RNG to produce identical results across replicas, since global
    # RNG would not be consistent across threads.
    rng = torch.Generator().manual_seed(2022 + step)

    cpu_inputs = torch.randn(global_batch_size, 10, generator=rng)
    cpu_labels = torch.randn(global_batch_size, 10, generator=rng)

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
    # To display the below messages, set '--debug'.
    # Here we don't use FLAGS.debug because this function is often ran in different processes than the launcher.
    if debug:
      print(
          f"iteration {step}: cpu_loss = {cpu_loss}, ddp_loss = {ddp_loss}, cpu_model.parameters() ~= ddp_model.parameters()"
      )
