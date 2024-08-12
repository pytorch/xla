import functools
import logging
import pickle
import jax
import numpy as np
import torch
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed._functional_collectives
from torch.nn.parallel import DistributedDataParallel as DDP_orig
import torch.distributed as dist
import torch.optim as optim
import torch_xla2
from jax.sharding import NamedSharding

from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import torch.utils._pytree as torch_pytree
from torch_xla2 import interop
from tqdm import tqdm


class DistributedDataParallel(torch.nn.Module):
  def __init__(self, module: torch.nn.Module, **kwargs):
    if kwargs:
      logging.warning(f'Unsupported kwargs {kwargs}')

    super().__init__()
    self._mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), axis_names=('batch',))
    replicated_state = torch_pytree.tree_map_only(torch.Tensor, lambda t: env.j2t_iso(jax.device_put(env.to_xla(t)._elem, NamedSharding(self._mesh, P()))), module.state_dict())
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
        arrays=[jax.device_put(env.to_xla(batch)._elem, device)
            for batch, device
            in zip(per_replica_batches, sharding.addressable_devices)])

  def replicate_input(self, inp):
    return env.j2t_iso(jax.device_put(inp._elem, NamedSharding(self._mesh, P())))

  def jit_step(self, func):

    @functools.wraps(func)
    def inner(*args):
      jax_states = self.state_dict()

      @interop.jax_jit
      def _jit_fn(states, args):
        self.load_state_dict(states)
        outputs = func(*args)
        return self.state_dict(), outputs

      new_states, outputs = _jit_fn(jax_states, args)
      self.load_state_dict(new_states)
      return outputs

    return inner

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
  print(jax.device_count(), 'devices')

  torch.manual_seed(0)
  # Create distributed data loader
  dataset = SortDataset('train')
  sampler = torch.utils.data.distributed.DistributedSampler(dataset, dist.get_world_size(), dist.get_rank(), shuffle=False)
  per_device_batch_size = 128
  local_batch_size = jax.local_device_count() * per_device_batch_size
  global_batch_size = jax.device_count() * per_device_batch_size
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=local_batch_size, sampler=sampler, drop_last=True)

  # Create model and wrap with DDP
  from mingpt.model import GPT
  def create_model():
    torch.manual_seed(0)
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = dataset.get_vocab_size()
    model_config.block_size = dataset.get_block_size()
    return GPT(model_config)

  jax_model = DistributedDataParallel(create_model())
  cpu_model = DDP_orig(create_model())

  for cp, jp in zip(cpu_model.parameters(), jax_model.parameters()):
    np.testing.assert_allclose(jp.detach().numpy(), cp.detach().numpy())

  # Define loss and optimizer
  cpu_optimizer = optim.SGD(cpu_model.parameters(), lr=3e-4)
  jax_optimizer = optim.SGD(jax_model.parameters(), lr=3e-4)

  # def step_fn(jax_data, jax_target):
  #   @interop.jax_jit
  #   def _step_fn(states, jax_data, jax_target):
  #     # jax_model.load_state_dict(states)
  #     jax_optimizer.zero_grad()
  #     jax_output, jax_loss = jax_model(jax_data, jax_target)
  #     jax_loss.backward()
  #     torch.nn.utils.clip_grad_norm_(jax_model.parameters(), 1.0)
  #     jax_optimizer.step()

  #     return jax_model.state_dict(), jax_output, jax_loss

  #   jax_states, jax_output, jax_loss = _step_fn(jax_model.state_dict(), jax_data, jax_target)
  #   jax_model.load_state_dict(jax_states)
  #   return jax_output, jax_loss

  @jax_model.jit_step
  def step_fn(jax_data, jax_target):
    jax_optimizer.zero_grad()
    jax_output, jax_loss = jax_model(jax_data, jax_target)
    jax_loss.backward()
    torch.nn.utils.clip_grad_norm_(jax_model.parameters(), 1.0)
    jax_optimizer.step()

  iters = 20000
  epochs = iters // global_batch_size + 1

  # Training loop
  for epoch in range(epochs):
    print('epoch', epoch)
    for data, target in tqdm(dataloader, unit='ex', unit_scale=global_batch_size):
      jax_data, jax_target = env.j2t_iso((jax_model.shard_input(data), jax_model.shard_input(target)))
      step_fn(jax_data, jax_target)

      # cpu_optimizer.zero_grad()
      # cpu_output, cpu_loss = cpu_model(data, target)
      # cpu_loss.backward()
      # cpu_optimizer.step()

  # TODO: this fails
  # for cp, jp in zip(cpu_model.parameters(), jax_model.parameters()):
  #   np.testing.assert_allclose(jp.detach().numpy(), cp.detach().numpy(), rtol=1.3e-6, atol=1e-5)

  input_cpu = torch.tensor([[0, 0, 2, 1, 0, 1]], dtype=torch.long)
  input_jax = env.to_xla(input_cpu)

  with torch.no_grad():
    cat_jax = jax_model._module.generate(jax_model.replicate_input(input_jax), input_jax[0].nelement(), do_sample=False)
    cat_cpu = cpu_model.module.generate(input_cpu, input_jax[0].nelement(), do_sample=False)

  sol_candidate_jax = cat_jax[:, input_jax.nelement():]
  sol_candidate_cpu = cat_cpu[:, input_cpu.nelement():]
  print('input sequence  :', input_cpu.tolist())
  print('predicted sorted (JAX):', sol_candidate_jax.numpy())
  print('predicted sorted (CPU):', sol_candidate_cpu.numpy())


if __name__ == '__main__':
  main()
