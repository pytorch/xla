import copy
import logging
import os
import pickle
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
from tqdm import tqdm

class DistributedDataParallel(torch.nn.Module):
  def __init__(self, module: torch.nn.Module, **kwargs):
    if kwargs:
      logging.warning(f'Unsupported kwargs {kwargs}')

    super().__init__()
    self._mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), axis_names=('batch',))
    replicated_state = torch_pytree.tree_map_only(torch.Tensor, lambda t: env.j2t_iso(jax.device_put(t.numpy(), NamedSharding(self._mesh, P()))), module.state_dict())
    # TODO: broadcast
    module.load_state_dict(replicated_state, assign=True)
    self._module = module

  def shard_input(self, inp):
    per_process_batch_size = inp.shape[0] # assumes batch dim is 0
    per_replica_batch_size = per_process_batch_size // jax.local_device_count()
    per_replica_batches = torch.chunk(inp, jax.local_device_count())
    global_batch_size = per_replica_batch_size * jax.device_count()
    global_batch_shape = ((global_batch_size,) + inp.shape[1:])

    sharding = NamedSharding(self._mesh, P('batch'))
    return jax.make_array_from_single_device_arrays(
        global_batch_shape,
        NamedSharding(self._mesh, P('batch')),
        arrays=[jax.device_put(batch.numpy(), device)
            for batch, device
            in zip(per_replica_batches, sharding.addressable_devices)])

  def forward(self, *args):
    return self._module(*args)

# from https://github.com/karpathy/minGPT/blob/master/demo.ipynb
class SortDataset(torch.utils.data.Dataset):
  """
  Dataset for the Sort problem. E.g. for problem length 6:
  Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
  Which will feed into the transformer concatenated as:
  input:  0 0 2 1 0 1 0 0 0 1 1
  output: I I I I I 0 0 0 1 1 2
  where I is "ignore", as the transformer is reading the input sequence
  """

  def __init__(self, split, length=6, num_digits=3):
    assert split in {'train', 'test'}
    self.split = split
    self.length = length
    self.num_digits = num_digits

  def __len__(self):
    return 10000 # ...

  def get_vocab_size(self):
    return self.num_digits

  def get_block_size(self):
    # the length of the sequence that will feed into transformer,
    # containing concatenated input and the output, but -1 because
    # the transformer starts making predictions at the last input element
    return self.length * 2 - 1

  def __getitem__(self, idx):
    # use rejection sampling to generate an input example from the desired split
    while True:
      # generate some random integers
      inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
      # half of the time let's try to boost the number of examples that
      # have a large number of repeats, as this is what the model seems to struggle
      # with later in training, and they are kind of rate
      if torch.rand(1).item() < 0.5:
          if inp.unique().nelement() > self.length // 2:
              # too many unqiue digits, re-sample
              continue
      # figure out if this generated example is train or test based on its hash
      h = hash(pickle.dumps(inp.tolist()))
      inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
      if inp_split == self.split:
          break # ok

    # solve the task: i.e. sort
    sol = torch.sort(inp)[0]

    # concatenate the problem specification and the solution
    cat = torch.cat((inp, sol), dim=0)

    # the inputs to the transformer will be the offset sequence
    x = cat[:-1].clone()
    y = cat[1:].clone()
    # we only want to predict at output locations, mask out the loss at the input locations
    y[:self.length-1] = -1
    return x, y

env = torch_xla2.default_env()

def main():
  dist.init_process_group(backend='gloo')
  # TODO: merge into backend
  # os.environ['TPU_VISIBLE_CHIPS'] = os.environ['LOCAL_RANK']
  # os.environ['TPU_PROCESS_BOUNDS'] = '1,1,1' if dist.get_world_size() == 1 else '2,2,1'
  # os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = '1,1,1'
  # ports = [str(p) for p in range(8476, 8480)]
  # os.environ['TPU_PROCESS_ADDRESSES'] = ','.join(f'localhost:{port}' for port in ports)
  # os.environ['TPU_PROCESS_PORT'] = ports[int(os.environ['LOCAL_RANK'])]
  # os.environ['CLOUD_TPU_TASK_ID'] = os.environ['RANK']

  # Create distributed data loader
  torch.manual_seed(0)
  dataset = SortDataset('train')
  sampler = torch.utils.data.distributed.DistributedSampler(dataset, dist.get_world_size(), dist.get_rank(), shuffle=False)
  per_device_batch_size = 2
  batch_size = jax.local_device_count() * per_device_batch_size
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

  # Create model and wrap with DDP
  from mingpt.model import GPT
  model_config = GPT.get_default_config()
  model_config.model_type = 'gpt-nano'
  model_config.vocab_size = dataset.get_vocab_size()
  model_config.block_size = dataset.get_block_size()
  model = GPT(model_config)
  # cpu_model = DDP_orig(copy.deepcopy(model))
  jax_model = DistributedDataParallel(model)

  # Define loss and optimizer
  # cpu_optimizer = optim.SGD(cpu_model.parameters(), lr=1)
  jax_optimizer = optim.SGD(jax_model.parameters(), lr=1)

  # Training loop
  for epoch in range(3):
    print('epoch', epoch)
    for data, target in tqdm(dataloader, unit='ex', unit_scale=batch_size):
      jax_data, jax_target = env.j2t_iso((jax_model.shard_input(data), jax_model.shard_input(target)))
      jax_optimizer.zero_grad()
      jax_output, jax_loss = jax_model(jax_data, jax_target)
      # jax_loss = loss_fn(jax_output, jax_target)
      jax_loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      jax_optimizer.step()

      # cpu_optimizer.zero_grad()
      # cpu_output, cpu_loss = cpu_model(data)
      # cpu_loss = loss_fn(cpu_output, target)
      # cpu_loss.backward()
      # cpu_optimizer.step()

      # for cp, jp in zip(cpu_model.parameters(), jax_model.parameters()):
      #  torch.testing.assert_close(jp, cp, check_device=False)

if __name__ == '__main__':
  main()
