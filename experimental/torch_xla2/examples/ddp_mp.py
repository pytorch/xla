import copy
import logging
import os
import jax
import torch
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed._functional_collectives
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP_orig
import torch.distributed as dist
import torch.optim as optim
import torch_xla2
from jax.sharding import NamedSharding

from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import torch.utils._pytree as torch_pytree

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

def main():
  dist.init_process_group(backend='gloo')
  # TODO: merge into backend
  os.environ['TPU_VISIBLE_CHIPS'] = os.environ['LOCAL_RANK']
  os.environ['TPU_PROCESS_BOUNDS'] = '2,2,1'
  os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = '1,1,1'
  ports = [str(p) for p in range(8476, 8480)]
  os.environ['TPU_PROCESS_ADDRESSES'] = ','.join(f'localhost:{port}' for port in ports)
  os.environ['TPU_PROCESS_PORT'] = ports[int(os.environ['LOCAL_RANK'])]
  os.environ['CLOUD_TPU_TASK_ID'] = os.environ['RANK']

  # Create distributed data loader
  torch.manual_seed(0)
  dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
  sampler = torch.utils.data.distributed.DistributedSampler(dataset, dist.get_world_size(), dist.get_rank(), shuffle=False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler, drop_last=True)

  # Create model and wrap with DDP
  model = nn.Linear(10, 1)
  cpu_model = DDP_orig(copy.deepcopy(model))
  jax_model = DistributedDataParallel(model)

  # Define loss and optimizer
  loss_fn = nn.MSELoss()
  cpu_optimizer = optim.SGD(cpu_model.parameters(), lr=1)
  jax_optimizer = optim.SGD(jax_model.parameters(), lr=1)

  # Training loop
  for epoch in range(3):
    print('epoch', epoch)
    for data, target in dataloader:
      data_shape, target_shape = data.numpy().shape, target.numpy().shape
      global_data_batch_shape = ((dataloader.batch_size * jax.process_count(),) + data_shape[1:])
      jax_data = jax.make_array_from_single_device_arrays(
        global_data_batch_shape,
        NamedSharding(jax_model._mesh, P('batch')),
        [jax.device_put(data.numpy(), jax.local_devices()[0])])
      global_target_batch_shape = ((dataloader.batch_size * jax.process_count(),) + target_shape[1:])
      jax_target = jax.make_array_from_single_device_arrays(
        global_target_batch_shape,
        NamedSharding(jax_model._mesh, P('batch')),
        [jax.device_put(target.numpy(), jax.local_devices()[0])])

      jax_data, jax_target = env.j2t_iso(jax_data), env.j2t_iso(jax_target)
      jax_optimizer.zero_grad()
      jax_output = jax_model(jax_data)
      jax_loss = loss_fn(jax_output, jax_target)
      jax_loss.backward()
      jax_optimizer.step()

      cpu_optimizer.zero_grad()
      cpu_output = cpu_model(data)
      cpu_loss = loss_fn(cpu_output, target)
      cpu_loss.backward()
      cpu_optimizer.step()

      for cp, jp in zip(cpu_model.parameters(), jax_model.parameters()):
        torch.testing.assert_close(jp, cp, check_device=False)

if __name__ == '__main__':
  main()
