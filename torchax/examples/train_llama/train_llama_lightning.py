import jax
import jax.numpy as jnp
from litgpt import config
from litgpt import model
from litgpt.data import Alpaca
from litgpt.tokenizer import Tokenizer
import lightning
import torch
from collections import defaultdict
from jax.experimental import shard_map

import torch.nn.functional
import torchax.interop

from . import utils
from . import model as editted_model
import os


def _setup_default_env():
  os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
  os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
  os.environ.setdefault('ALLOW_MULTIPLE_LIBTPU_LOAD', '1')
  # only need for tpu v4
  # os.environ.setdefault('TPU_MEGACORE', 'megacore_dense')
  tpu_args = "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

  os.environ.setdefault('LIBTPU_INIT_ARGS', tpu_args)


_setup_default_env()

default_checkpoint_dir = '/home/hanq/litgpt/checkpoints/meta-llama/Meta-Llama-3-8B/'


class GPTLightningModule(lightning.LightningModule):

  def __init__(self, gpt):
    super().__init__()
    self.gpt = utils.FSDPv2(gpt)

  def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self.gpt.forward(x)
    num_tokens = logits.shape[-1]
    logits = logits[..., :-1, :].reshape(-1, num_tokens)
    y = y[..., 1:].reshape(-1)
    return torch.nn.functional.cross_entropy(logits, y)

  def configure_optimizers(self):
    return None


from jax.experimental import mesh_utils

P = jax.sharding.PartitionSpec
mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh(utils.num_partitions),
    axis_names=utils.global_axis,
)


class GPTOutline(torch.nn.Module):

  def __init__(self, gpt_orig):
    super().__init__()
    self.gpt_orig = gpt_orig

    def one_layer(weights, args):
      return torch.func.functional_call(self.gpt_orig.transformer.h[0], weights,
                                        args)

    self.one_layer = torchax.interop.jax_jit(one_layer)

  def forward(self, idx: torch.Tensor, input_pos=None) -> torch.Tensor:
    T = idx.size(1)
    if self.gpt_orig.max_seq_length < T:
      raise ValueError(
          f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
      )

    cos = self.gpt_orig.cos[:T]
    sin = self.gpt_orig.sin[:T]
    mask = None

    x = self.gpt_orig.transformer.wte(
        idx)  # token embeddings of shape (b, t, n_embd)
    editted_model.reapply_sharding(x)
    if self.gpt_orig.config.scale_embeddings:
      x = x * (self.gpt_orig.config.n_embd**0.5)

    for block in self.gpt_orig.transformer.h:
      args = (x, cos, sin, mask, input_pos)
      weights = block.state_dict()
      x = self.one_layer(weights, args)
      editted_model.reapply_sharding(x)

    x = self.gpt_orig.transformer.ln_f(x)
    editted_model.reapply_sharding(x)
    res = self.gpt_orig.lm_head(x)  # (b, t, vocab_size)
    editted_model.reapply_sharding(res)
    return res


class GPTFori:

  def __init__(self, gpt_orig, manual_all_gather=False):
    super().__init__()
    self.gpt_orig = gpt_orig

    one_block = self.gpt_orig.transformer.h[0]
    self.manual_all_gather = manual_all_gather

    def one_layer(args, weights):
      # inputs are jax array
      orig_args = args

      x, cos, sin, mask, input_pos = args
      if self.manual_all_gather:
        weights, cos, sin = jax.lax.all_gather((weights, cos, sin),
                                               'fsdp',
                                               tiled=True)
      args = (x, cos, sin, mask, input_pos)
      args, weights = torchax.default_env().j2t_iso((args, weights))
      res = torch.func.functional_call(one_block, weights, args)
      res = torchax.default_env().t2j_iso(res)
      return (res, *orig_args[1:]), jnp.array([0])

    if self.manual_all_gather:
      one_layer = shard_map.shard_map(
          one_layer,
          mesh=mesh,
          in_specs=(P(*utils.global_axis), P(*utils.global_axis)),
          out_specs=(P(*utils.global_axis), P()),
          check_rep=False)

    one_layer = jax.checkpoint(
        one_layer,
        policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)

    def compiled_block(weights, x):
      x, _ = jax.lax.scan(one_layer, x, weights, unroll=4)
      return x[0]

    self.compiled_block = compiled_block

    self.weights = {
        'wte': self.gpt_orig.transformer.wte.state_dict(),
        'block': self.make_weights_scan(),
        'ln_f': self.gpt_orig.transformer.ln_f.state_dict(),
        'lm_head': self.gpt_orig.lm_head.state_dict(),
        'sin': self.gpt_orig.sin,
        'cos': self.gpt_orig.cos,
    }

  def make_weights_scan(self):
    temp = defaultdict(list)  # key to list of tensors
    for block in self.gpt_orig.transformer.h:
      state_dict = block.state_dict()
      for k, v in state_dict.items():
        temp[k].append(v)

    temp = {k: torch.stack(v) for k, v in temp.items()}
    return temp

  def forward_with_weights(self, weights, idx):
    T = idx.size(1)
    if self.gpt_orig.max_seq_length < T:
      raise ValueError(
          f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
      )

    cos = weights['cos'][:T]
    sin = weights['sin'][:T]
    mask = None

    x = torch.func.functional_call(
        self.gpt_orig.transformer.wte,
        weights['wte'],
        idx,
    )

    editted_model.reapply_sharding(x)

    if self.gpt_orig.config.scale_embeddings:
      x = x * (self.config.n_embd**0.5)
    editted_model.reapply_sharding(x)

    args = (x, cos, sin, mask, None)
    #import pdb; pdb.set_trace()
    x = torchax.interop.call_jax(
        self.compiled_block,
        weights['block'],
        args,
    )
    editted_model.reapply_sharding(x)

    x = torch.func.functional_call(self.gpt_orig.transformer.ln_f,
                                   weights['ln_f'], x)
    editted_model.reapply_sharding(x)
    x = torch.func.functional_call(self.gpt_orig.lm_head, weights['lm_head'], x)
    editted_model.reapply_sharding(x)
    return x


import logging
import torchax

# Modes:

REGULAR = 'regular'
OUTLINED_LAYER = 'outlined'  # jax.jit a block to get faster compilation
SCAN_LAYER = 'scan_layer'  # jax.lax.scan for looping layers to get faster compilation
SCAN_LAYER_MANUAL = 'scan_manual'  # jax.lax.scan AND shmap layers to get faster compilation


def main_one(
    use_flash_attention=True,
    seqlen=8196,
    n_layers=32,
    batch_size=8,
    checkpoint_dir=default_checkpoint_dir,
    mode='regular',
    use_editted_model=False,
):
  logging.getLogger("jax").setLevel(logging.DEBUG)
  print(f"Running with parameters {locals()}")
  utils.SEQLEN = seqlen
  utils.BATCH = batch_size
  env = torchax.default_env()
  env.config.use_tpu_flash_attention = use_flash_attention
  cfg = config.Config.from_name("Meta-Llama-3-8B")
  cfg.n_layer = n_layers
  #cfg.n_layer = 32
  if use_editted_model:
    gpt = editted_model.GPT(cfg)
  else:
    gpt = model.GPT(cfg)
  gpt.to(torch.bfloat16)

  env.config.shmap_flash_attention = mode != SCAN_LAYER_MANUAL
  use_fori = False

  if mode in (SCAN_LAYER, SCAN_LAYER_MANUAL):
    gpt = GPTFori(gpt, mode == SCAN_LAYER_MANUAL)
    use_fori = True
  elif mode == OUTLINED_LAYER:
    gpt = GPTOutline(gpt)

  light_mod = GPTLightningModule(gpt)
  tokenizer = Tokenizer(checkpoint_dir)
  data = Alpaca(num_workers=1)
  data.connect(
      tokenizer=tokenizer, batch_size=batch_size, max_seq_length=utils.SEQLEN)
  data.prepare_data()
  data.setup()
  train_loader = data.train_dataloader()

  with mesh:
    trainer = utils.JaxTrainer(use_fori)
    if use_fori:
      return trainer.fit_model_fori(gpt, train_loader)
    else:
      return trainer.fit(light_mod, train_loader)


def main(
    use_flash_attention=True,
    seqlen=8196,
    n_layers=32,
    batch_size=8,
    checkpoint_dir=default_checkpoint_dir,
    mode='regular',
    use_editted_model=False,
):
  if mode == 'all':
    from jaxlib.xla_extension import XlaRuntimeError
    res = []
    for m, editted in ((SCAN_LAYER, False), (SCAN_LAYER_MANUAL, False),
                       (REGULAR, False), (REGULAR, True), (OUTLINED_LAYER,
                                                           True)):
      try:
        run_time, comp_time = main_one(
            use_flash_attention,
            seqlen,
            n_layers,
            batch_size,
            checkpoint_dir,
            m,
            use_editted_model=editted,
        )
        res.append((m, editted, run_time, comp_time))
      except XlaRuntimeError as e:
        import traceback
        traceback.print_exc()
        res.append((m, editted, 'OOM', ''))
    for m, e, r, c in res:
      print(f'{m}-edit={e}: \t {r} \t {c} ')

  else:
    main_one(use_flash_attention, seqlen, n_layers, batch_size, checkpoint_dir,
             mode, use_editted_model)


if __name__ == '__main__':
  import fire
  fire.Fire(main)
