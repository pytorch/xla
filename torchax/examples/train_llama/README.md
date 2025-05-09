Lightning Based training for llama 3
====================================

# Abstract:
We train llama3 model using a the PyTorch implementation from [litgpt]().
We train it using a pytorch-lightning like setup, where user defines a `training_step` methods. The trainer itself is a custom trainer that works on CloudTPU by leverying jax.

## Result:

* Best v5p MFU: 63.7% (batch = 8, sequence length = 8192)
* Best v4-8 MFU: 44% (batch = 8, sequence length = 2048)

# Setup:

```bash
pip install 'litgpt[all]' optax fire

litgpt  download  meta-llama/Meta-Llama-3-8B-Instruct
litgpt  download  --repo_id meta-llama/Meta-Llama-3-8B-Instruct --tokenizer_only true --access_token <your_hf_token>
```

Then, run with
```bash
python -m examples.train_llama.train_llama_lightning  --mode=all --seqlen=2048 --checkpoint_dir=<dir where the checkpoint is downloaded>
```

# The training script

The script in [train_llama_lightning.py](train_llama_training.py) is envisioned
to be what the users need to write for their training setup. In summary it
consists the following:

### 1. Put the model in a `LightningModule` subclass:

```python
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
        return torch.nn.functional.cross_entropy(
            logits, y)

    def configure_optimizers(self):
        return None
```

This class is responsible of wrapping (or instantiating) a model, call it's forward, and defining the formula to compute loss.

Next, the user need to call our trainer along with the dataloader:

```python
def main():
    gpt = ...
    light_mod = GPTLightningModule(gpt)
    # data loader setup and stuff skipped
    train_loader = ...

    # Train
    trainer = ...
    trainer.fit(light_mod, train_loader)
```

The actual script in train_llama_lightning.py is more complex because
we are testing out different options and optimizing strategies.

The trainer itself, as well as the helper class for sharding strategy (FSDPv2),
is defined in [utils.py](utils.py). In the future, we hope to upstream these
into pytorch-lightning and becames one of the `Strategy` that `pl.Trainer` uses.

### FSDPv2

FSDPv2 is an implementation of Fully-sharded Data Parallel training strategy using
GSPMD. To implement this, we need 2 things:

1. Shard inputs on batch dimension (i.e. like DDP)
2. Shard all the weights in the first dimension.

To implement this, we create a mesh with first axis called 'fsdp' and shard
everything on this.

```python
class FSDPv2(torch.nn.Module):

    def __init__(self, mod):
        super().__init__()
        self.mod = mod

        num_of_partitions = jax.device_count()
        self.mesh = jax.sharding.Mesh(
            mesh_utils.create_device_mesh((num_of_partitions, )),
            axis_names=("fsdp", ),
        )
        self.sharding = jax.sharding.NamedSharding(self.mesh, P("fsdp"))

    def forward(self, *args):
        args = list(args)
        args[0] = self.shard(args[0])
        res = self.mod(*args)
        return self.shard(res)

    def shard(self, x):
        return torchax.interop.call_jax(
            jax.lax.with_sharding_constraint,
            x,
            self.sharding,
        )
```
We also need a similar function that shards the weights.

### Flash attention

Flash attention is a important optimization that enables training with large
sequence length. Jax has an implementation of flash attention located in
`jax.experimental.pallas.ops.tpu.flash_attention`. To make the model uses
this version of flash attention, we simply register a lowering for PyTorch's
`torch.nn.functional.scaled_dot_product_attention` like so:

```python
@register_function(torch.nn.functional.scaled_dot_product_attention, is_jax_function=False, needs_env=True)
def scaled_dot_product_attention(
   query, key, value, attn_mask=None,
   dropout_p=0.0, is_causal=False, scale=None, env=None) -> torch.Tensor:

   if env.use_flash_attention:
    jquery, jkey, jvalue = env.t2j_iso((query, key, value))
    res = _tpu_flash_attention(jquery, jkey, jvalue, env)
    return env.j2t_iso(res)

   return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale)
```

this implementation is located in [jtorch.py](../../torchax/ops/jtorch.py) in
torchax. The model itself does not need to change to use TPU version of
flash attention, because it's calling pytorch's `F.scaled_dot_product_attention`.

## Misc optimizations


### Compile one layer first

A program compiled with `jax.jit` is a straight graph of XLA operators (StableHLO ops). For the llama3 model, it consists of 32 layers of identical code. This makes compile time extremely long. We can outline one of the layers, and call that one repeatedly to get slightly faster compile time.

To compile the regular 32-layer model, it takes 210s on v4-8; with outlining we
can reduce this to 190s. And program size (number of chars in `jax.jit(...).lowered.as_text()`) is reduced from 2.6 million to 1.9 million.

To accomplish this, we can wrap the original model with the `GPTOutline` wrapper.

### Use `jax.lax.scan` to iterate the layers

Another more intrusive change is to wrap to change the loop to use
scan instead of python loop. This way we can get an even smaller program.
This change is illustrated in `GPTFori` wrapper.
With this change, we can shrink the program size to 0.49 million characters, and
compile time to 19.8s.

Scan makes compiling faster but makes runtime slightly slower: XLA's ability
of optimizing across control flow boundaries is less than it's ability to optimize
on straight graph, so we lose a bit of runtime perf.


## Detailed numbers

### v5p-8

seqlen = 8192
bs = 8

| Batch Size   | Sequence Length   | Mode                  | Step Time (s)   | Compile Time  (s) | MFU    |
|:-------------|:------------------|:-----------------------|:------------|:---------------|:-------------|
| 8            | 8192              | scan_layer  | 4.38044     | 12.22        | 49.76%        |
| 8            | 8192              | scan_manual | 4.32437     | 12.95        | 50.41%        |
| 8            | 8192              | regular     | 4.56214     | 1086.71        | 47.78%        |
| 8            | 8192              | outlined    | 3.41887     | 1079.77        | 63.76%        |


### v4-8

seqlen = 2048
bs = 8

| Batch Size   | Sequence Length   | Mode        | Step Time  (s)  | Compile Time (s) | MFU   |
|:-------------|:------------------|:------------|:-------------|:---------------|:------|
| 8            | 2048              | scan_layer  | 1.80099      | 17.61        | 42%   |
| 8            | 2048              | scan_manual | 1.85214      | 16.69        | 41%   |
| 8            | 2048              | regular     | 1.70979      | 362.32        | 44%   |
| 8            | 2048              | outlined    | OOM          | -              | -     |
