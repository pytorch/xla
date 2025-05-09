"""WIP example using `minGPT` with DistributedDataParallel on both CPU and JAX.

Required `mingpt` package for model definition (see requirements.txt). Some
hyperparameters and training configuration borrowed from nanoGPT:
https://github.com/karpathy/nanoGPT

Example command (single host):
torchrun --standalone xla/experimental/torchax/examples/train_gpt/train_ddp.py

Tested on a TPU v4-8
"""

import datetime
import jax
import torch
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import torch.optim as optim
import torchax
from tqdm import tqdm
from mingpt.model import GPT
from datasets import load_dataset
import tiktoken
import pathlib
import torch.utils._pytree as torch_pytree


def _checkpoint(jax_model, path: pathlib.Path):
  torch.save(
      torch_pytree.tree_map_only(
          torchax.tensor.Tensor,
          torchax.tensor.Tensor.torch,
          jax_model.state_dict(),
      ),
      path,
  )


def main():
  dist.init_process_group(backend="gloo")
  dataset_name = "Skylion007/openwebtext"
  dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)

  enc = tiktoken.get_encoding("gpt2")

  def tokenize(ex):
    """Tokenize each example and append the end-of-text token."""
    ids = enc.encode_ordinary(ex["text"])
    ids.append(enc.eot_token)
    return {"ids": ids}

  dataset = dataset.map(tokenize, num_proc=16)

  def group_texts(exs):
    """Group batches of tokens into `block_size` chunks."""
    cat = torch.cat([torch.tensor(ex) for ex in exs["ids"]])
    total_len = cat.size()[0]
    num_chunks = total_len // 1025
    split = torch.split(cat[:num_chunks * 1025], 1025)
    xs = [ex[:-1] for ex in split]
    ys = [ex[1:] for ex in split]
    return {"x": xs, "y": ys}

  dataset = dataset.map(
      group_texts, batched=True, remove_columns=["text", "ids"], num_proc=16)
  dataset.shard(dist.get_world_size(), dist.get_rank())
  env = torchax.default_env()

  print(jax.device_count(), "devices")

  torch.manual_seed(0)
  per_device_batch_size = 8
  local_batch_size = jax.local_device_count() * per_device_batch_size
  global_batch_size = jax.device_count() * per_device_batch_size
  dataloader = torch.utils.data.DataLoader(
      dataset.with_format("torch"), batch_size=local_batch_size, drop_last=True)

  # Create model and wrap with DDP
  def create_model():
    torch.manual_seed(0)
    model_config = GPT.get_default_config()
    model_config.model_type = "gpt2"
    model_config.vocab_size = enc.n_vocab
    model_config.block_size = 1024
    # TODO: use bf16 when erroneous type promotions are fixed
    return GPT(model_config)  # .to(dtype=torch.bfloat16)

  checkpoint_subdir = pathlib.Path(
      "checkpoints") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  checkpoint_subdir.mkdir(parents=True)
  jax_model = torchax.distributed.DistributedDataParallel(create_model(), env)

  # TODO: LR scheduler
  jax_optimizer = optim.SGD(jax_model.parameters(), lr=6e-4, weight_decay=0.1)

  # Contents of `step_fn` can be inlined if using eager
  @jax_model.jit_step
  def step_fn(jax_data, jax_target):
    jax_optimizer.zero_grad()
    jax_output, jax_loss = jax_model(jax_data, jax_target)
    jax_loss.backward()
    torch.nn.utils.clip_grad_norm_(jax_model.parameters(), 1.0)
    jax_optimizer.step()

    return jax_output, jax_loss

  tokens_per_batch = global_batch_size * 1024

  for epoch in range(1):
    print("epoch", epoch)
    for i, batch in enumerate(
        tqdm(dataloader, unit="tok", unit_scale=tokens_per_batch)):
      data, target = batch["x"], batch["y"]
      jax_data, jax_target = env.j2t_iso(
          (jax_model.shard_input(data), jax_model.shard_input(target)))
      jax_output, jax_loss = step_fn(jax_data, jax_target)

      if i % 1000 == 0:
        _checkpoint(jax_model, checkpoint_subdir / "gpt2_124m_{epoch}_{i}.ckpt")
        print("step", i, jax_loss.item())

  with torch.no_grad():
    with env:
      inp = enc.encode("This GPT-2 example is")
      input_jax = torch.tensor([inp], dtype=torch.long)
      # TODO: need to access underlying module for methods
      jax_generated = jax_model._module.generate(
          jax_model.replicate_input(input_jax),
          100,
          do_sample=False,
      )

  print("input sequence:", inp, enc.decode(inp))
  print(jax_generated)
  print("predicted (JAX):", enc.decode(jax_generated.numpy().tolist()))


if __name__ == "__main__":
  main()
