"""Example using `minGPT` with DistributedDataParallel on both CPU and JAX.

Example command (single host):
OMP_NUM_THREADS=16 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12355 python xla/experimental/torch_xla2/examples/train_g
pt/train_ddp.py
"""

import pickle
import jax
import numpy as np
import torch
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import torch.optim as optim
import torch_xla2
from tqdm import tqdm
from mingpt.model import GPT


# Dataset copied from `minGPT` demo:
# https://github.com/karpathy/minGPT/blob/master/demo.ipynb
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
    assert split in {"train", "test"}
    self.split = split
    self.length = length
    self.num_digits = num_digits

  def __len__(self):
    return 10000  # ...

  def get_vocab_size(self):
    return self.num_digits

  def get_block_size(self):
    return self.length * 2 - 1

  def __getitem__(self, idx):
    while True:
      inp = torch.randint(
        self.num_digits, size=(self.length,), dtype=torch.long
      )
      if torch.rand(1).item() < 0.5:
        if inp.unique().nelement() > self.length // 2:
          continue
      h = hash(pickle.dumps(inp.tolist()))
      inp_split = "test" if h % 4 == 0 else "train"
      if inp_split == self.split:
        break

    sol = torch.sort(inp)[0]

    cat = torch.cat((inp, sol), dim=0)

    x = cat[:-1].clone()
    y = cat[1:].clone()
    y[: self.length - 1] = -1
    return x, y


def main():
  env = torch_xla2.default_env()

  dist.init_process_group(backend="gloo")
  print(jax.device_count(), "devices")

  torch.manual_seed(0)
  dataset = SortDataset("train")
  sampler = torch.utils.data.distributed.DistributedSampler(
    dataset, dist.get_world_size(), dist.get_rank(), shuffle=False
  )
  per_device_batch_size = 128
  local_batch_size = jax.local_device_count() * per_device_batch_size
  global_batch_size = jax.device_count() * per_device_batch_size
  dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=local_batch_size, sampler=sampler, drop_last=True
  )

  # Create model and wrap with DDP
  def create_model():
    torch.manual_seed(0)
    model_config = GPT.get_default_config()
    model_config.model_type = "gpt-nano"
    model_config.vocab_size = dataset.get_vocab_size()
    model_config.block_size = dataset.get_block_size()
    return GPT(model_config)

  jax_model = torch_xla2.distributed.DistributedDataParallel(
    create_model(), env
  )
  cpu_model = torch.nn.parallel.DistributedDataParallel(create_model())

  # Check that models initialized to same parameters
  for cp, jp in zip(cpu_model.parameters(), jax_model.parameters()):
    np.testing.assert_allclose(jp.detach().numpy(), cp.detach().numpy())

  cpu_optimizer = optim.SGD(cpu_model.parameters(), lr=3e-4)
  jax_optimizer = optim.SGD(jax_model.parameters(), lr=3e-4)

  # Contents of `step_fn` can be inlined if using eager
  # TODO: JIT is slow
  # @jax_model.jit_step
  def step_fn(jax_data, jax_target):
    jax_optimizer.zero_grad()
    jax_output, jax_loss = jax_model(jax_data, jax_target)
    jax_loss.backward()
    torch.nn.utils.clip_grad_norm_(jax_model.parameters(), 1.0)
    jax_optimizer.step()

    return jax_output, jax_loss

  iters = 20000
  epochs = iters // global_batch_size + 1

  for epoch in range(epochs):
    print("epoch", epoch)
    for data, target in tqdm(
      dataloader, unit="ex", unit_scale=global_batch_size
    ):
      jax_data, jax_target = env.j2t_iso(
        (jax_model.shard_input(data), jax_model.shard_input(target))
      )
      jax_output, jax_loss = step_fn(jax_data, jax_target)

      cpu_optimizer.zero_grad()
      cpu_output, cpu_loss = cpu_model(data, target)
      cpu_loss.backward()
      torch.nn.utils.clip_grad_norm_(cpu_model.parameters(), 1.0)
      cpu_optimizer.step()

      # TODO: this fails, even without DDP
      # for cp, jp in zip(cpu_model.parameters(), jax_model.parameters()):
      #   np.testing.assert_allclose(jax_loss.item(), cpu_loss.item(), rtol=1.3e-6, atol=1e-5)

    print("jax loss", jax_loss.item())
    print("cpu loss", cpu_loss.item())

  with torch.no_grad():
    with env:
      input_jax = torch.tensor([[0, 0, 2, 1, 0, 1]], dtype=torch.long)
      # TODO: need to access underlying module for methods
      cat_jax = jax_model._module.generate(
        jax_model.replicate_input(input_jax),
        input_jax[0].nelement(),
        do_sample=False,
      )

    input_cpu = torch.tensor([[0, 0, 2, 1, 0, 1]], dtype=torch.long)
    cat_cpu = cpu_model.module.generate(
      input_cpu, input_jax[0].nelement(), do_sample=False
    )

  sol_candidate_jax = cat_jax[:, input_jax.nelement() :]
  sol_candidate_cpu = cat_cpu[:, input_cpu.nelement() :]
  print("input sequence  :", input_cpu.tolist())
  print("predicted sorted (JAX):", sol_candidate_jax.numpy())
  print("predicted sorted (CPU):", sol_candidate_cpu.numpy())


if __name__ == "__main__":
  main()
