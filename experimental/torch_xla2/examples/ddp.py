import copy
import logging
import jax
import torch
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed._functional_collectives
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch_xla2
from jax.sharding import NamedSharding

from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import torch.utils._pytree as torch_pytree

# Initialize distributed communication
dist.init_process_group(backend='jax', init_method='jax://')
rank = dist.get_rank()

class DistributedDataParallel(torch.nn.Module):
  def __init__(self, module: torch.nn.Module, **kwargs):
    if kwargs:
      logging.warning(f'Unsupported kwargs {kwargs}')

    super().__init__()
    self._mesh = Mesh(mesh_utils.create_device_mesh((4,)), axis_names=('batch',))
    replicated_state = torch_pytree.tree_map_only(torch.Tensor, lambda t: env.j2t_iso(jax.device_put(t.numpy(), NamedSharding(self._mesh, P()))), module.state_dict())
    # TODO: broadcast
    module.load_state_dict(replicated_state, assign=True)
    self._module = module

  def forward(self, *args):
    return self._module(*args)

env = torch_xla2.default_env()

# Create model and wrap with DDP
model = nn.Linear(10, 1)
cpu_model = copy.deepcopy(model)
jax_model = DistributedDataParallel(model)

# Create distributed data loader
dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, sampler=sampler)

# Define loss and optimizer
loss_fn = nn.MSELoss()
cpu_optimizer = optim.SGD(cpu_model.parameters(), lr=1)
jax_optimizer = optim.SGD(jax_model.parameters(), lr=1)

# Training loop
for epoch in range(3):
    print('epoch', epoch)
    for data, target in dataloader:
        cpu_optimizer.zero_grad()
        cpu_output = cpu_model(data)
        cpu_loss = loss_fn(cpu_output, target)
        cpu_loss.backward()
        cpu_optimizer.step()

        jax_data, jax_target = env.j2t_iso(jax.device_put([data.numpy(), target.numpy()], NamedSharding(jax_model._mesh, P('batch'))))
        jax_optimizer.zero_grad()
        jax_output = jax_model(jax_data)
        jax_loss = loss_fn(jax_output, jax_target)
        jax_loss.backward()
        jax_optimizer.step()

        for cp, jp in zip(cpu_model.parameters(), jax_model.parameters()):
          torch.testing.assert_close(jp, cp, check_device=False)
