import functools
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
from jax.experimental.shard_map import shard_map

# Initialize distributed communication
dist.init_process_group(backend='jax', init_method='jax://')
rank = dist.get_rank()

class DistributedDataParallel(torch.nn.Module):
  def __init__(self, module: torch.nn.Module, **kwargs):
    if kwargs:
      logging.warning(f'Unsupported kwargs {kwargs}')

    # Does this make sense? https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook
    # TODO broadcast and/or check params
    for p in module.parameters():
      if p.requires_grad:
        p.register_hook(lambda grad: print('backward hook') or torch.distributed._functional_collectives.all_reduce(grad, 'sum', ''))

    super().__init__()
    self._module = module
    self._mesh = Mesh(mesh_utils.create_device_mesh((4,)), axis_names=('torch_dist',))

  def forward(self, *args):
    return self._module(*args)
    # @functools.partial(
    #       shard_map,
    #       mesh=self._mesh,
    #       in_specs=P('torch_dist'),
    #       out_specs=P('torch_dist'))
    # def jax_wrapper(jax_args):
    #     args = env.j2t_iso(jax_args)
    #     torch_outputs = self._module(*args)
    #     return env.t2j_iso(torch_outputs)

    # jax_outputs = jax_wrapper(env.t2j_iso(args))
    # return env.j2t_iso(jax_outputs) # TODO: output has no grad


env = torch_xla2.default_env()

# Create model and wrap with DDP
with env:
  model = nn.Linear(10, 1)
model = DistributedDataParallel(model)


# Create distributed data loader
dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, sampler=sampler)

# Define loss and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1)


@functools.partial(
      shard_map,
      mesh=model._mesh,
      in_specs=P('torch_dist'),
      out_specs=P())
def step_fn(data, target):
    data, target = env.j2t_iso((data, target))
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    return jax.lax.psum(loss._elem, 'torch_dist')

# Training loop
for epoch in range(3):
    print(epoch)
    for data, target in dataloader:
        data, target = jax.device_put([data.numpy(), target.numpy()], NamedSharding(model._mesh, P('torch_dist')))
        jax_loss = step_fn(data, target)
        print(jax_loss)
        print([p.grad for p in model.parameters()])
        print([p.mean().numpy() for p in model.parameters()])
