Training based on torchtitan llama model
====================================

This examples demonstrates how we can make a model implemented for single device
run on multiple devices without modifying the model itself.

We choose [torchtitan's llama implementation](https://github.com/pytorch/torchtitan/tree/main/torchtitan/models/llama);
because torchtitan's model implementation is a clean single device version. (Not those
sprinkled with `ColumnParallelLinear`'s from megatron). torchtitan accomplishes running
single device model code in multi-device environment through module-swaps, and we accomplishes
the same with gSPMD.



## Install dependencies

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install optax fire tensorflow tensorboard-plugin-profile
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

cd ~
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
pip install -r requirements.txt
pip install .

cd ~
git clone https://github.com/pytorch/xla.git
cd xla/experimental/torchax
pip install -e .
```

(Optional) Export libtpu flags that helps with performance
```bash
export LIBTPU_INIT_ARGS="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
```
NOTE: these flags are copied from https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/configs/trillium/llama2_70b_4096.sh
Tested locally on v6e-8 doesnt seems to make a difference.

```bash
cd ~/xla/experimental/torchax/examples/train_llama_torchtitan
python train_llama.py --seqlen=8192
```

## Detailed Code walkthrough:

Below is the copy & paste of `train_llama.py` and annotated with what they do:

```python
import os
import time
import logging
from typing import Tuple
from collections import defaultdict
import functools
import torch
import torch.nn.functional
from torch.utils import _pytree as pytree
import splash_attn
import helper

import torchax as tx
import torchax.interop
import torchax.train
from torchax.interop import jax_view, torch_view, JittableModule
import jax
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding
import optax
```
Above is just regular imports, uninteresting

```python
from torchtitan.models.llama import llama3_configs
from torchtitan.models.llama import model as titan
```
Above is importing the model and model config from torchtitan directly.
i.e. we don't need to modify the model code at all (note, there are caveats, keep reading).

```python
P = jax.sharding.PartitionSpec
num_global_devices = jax.device_count()
num_local_devices = jax.local_device_count()
```
This bit above are some aliases


```python
def sharded_device_put(tensor: jax.Array, sharding) -> jax.Array:
    if isinstance(tensor, tuple):
        return tuple(sharded_device_put(t, sharding) for t in tensor)

    if num_global_devices == num_local_devices:
        return jax.device_put(tensor, sharding)

    # NOTE: at here, num_global_devices != num_local_devices
    # meaning we are in multi-host setup. Each host will run the same process
    # and each process only need to handle the devices accessible to this host.
    shape = tensor.shape
    x_split = [jax.device_put(tensor[i], device)
               for device, i in sharding.addressable_devices_indices_map(shape).items()]
    return jax.make_array_from_single_device_arrays(shape, sharding, x_split)
```

When running on single-host, `jax.device_put` suffices. Multi-host need some
extra incantations so that we split an array to only the shards corresponding
to the accessible devices in this host.


```python
sharding_map_original = {
  "freqs_cis" : (), #  torch.complex64 (2048, 64)
  "tok_embeddings.weight" : ('fsdp', 'tp'), #  torch.float32 (vocab_size, 4096)
  "layers.*.attention.wo.weight" : ('fsdp', 'tp'), #  torch.int8 (4096, 4096)
  "layers.*.attention.wq.weight" : ('tp', 'fsdp'), #  torch.int8 (4096, 4096)
  "layers.*.attention.wk.weight" : ('tp', 'fsdp'), #  torch.int8 (4096, 4096)
  "layers.*.attention.wv.weight" : ('tp', 'fsdp'), #  torch.int8 (4096, 4096)
  "layers.*.feed_forward.w1.weight" : ('tp', 'fsdp'), #  torch.float32 (11008, 4096)
  "layers.*.feed_forward.w2.weight" : ('fsdp', 'tp'), #  torch.float32 (4096, 11008)
  "layers.*.feed_forward.w3.weight": ('tp', 'fsdp'), #  torch.float32 (11008, 4096)
  "layers.*.attention_norm.weight" : ('fsdp', ), #  torch.float32 (4096,)
  "layers.*.ffn_norm.weight" : ('fsdp', ), #  torch.float32 (4096,)
  "norm.weight" : ('fsdp', ), #  torch.float32 (4096,)
  "output.weight" : ('tp', 'fsdp'), #  torch.float32 (vocab_size, 4096)
}

sharding_map_scan = {
  "freqs_cis" : (), #  torch.complex64 (2048, 64)
  # ParallelEmbedding for llama2; VocabParallelEmbedding for 3
  "tok_embeddings.weight" : ('tp', 'fsdp'), #  torch.float32 (vocab_size, 4096)
  "layers.params.attention___wo___weight" : (None, 'fsdp', 'tp'), #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wq___weight" : (None, 'tp', 'fsdp'), #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wk___weight" : (None, 'tp', 'fsdp'), #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wv___weight" : (None, 'tp', 'fsdp'), #  torch.int8 (n, 4096, 4096)
  "layers.params.feed_forward___w1___weight" : (None, 'tp', 'fsdp'), #  torch.float32 (n, 11008, 4096)
  "layers.params.feed_forward___w2___weight" : (None, 'fsdp', 'tp'), #  torch.float32 (n, 4096, 11008)
  "layers.params.feed_forward___w3___weight": (None, 'tp', 'fsdp'), #  torch.float32 (n, 11008, 4096)
  "layers.params.attention_norm___weight" : (None, 'fsdp', ), #  torch.float32 (n, 4096,)
  "layers.params.ffn_norm___weight" : (None, 'fsdp', ), #  torch.float32 (n, 4096,)
  "norm.weight" : ('fsdp', ), #  torch.float32 (4096,)
  "output.weight" : ('tp', 'fsdp'), #  torch.float32 (vocab_size, 4096)
}

sharding_map_scan_fsdp = {
  "freqs_cis" : (), #  torch.complex64 (2048, 64)
  # ParallelEmbedding for llama2; VocabParallelEmbedding for 3
  "tok_embeddings.weight" : ('fsdp',), #  torch.float32 (vocab_size, 4096)
  "layers.params.attention___wo___weight" : (None, 'fsdp'), #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wq___weight" : (None, 'fsdp'), #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wk___weight" : (None, 'fsdp'), #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wv___weight" : (None, 'fsdp'), #  torch.int8 (n, 4096, 4096)
  "layers.params.feed_forward___w1___weight" : (None, 'fsdp'), #  torch.float32 (n, 11008, 4096)
  "layers.params.feed_forward___w2___weight" : (None, 'fsdp'), #  torch.float32 (n, 4096, 11008)
  "layers.params.feed_forward___w3___weight": (None, 'fsdp'), #  torch.float32 (n, 11008, 4096)
  "layers.params.attention_norm___weight" : (None, 'fsdp', ), #  torch.float32 (n, 4096,)
  "layers.params.ffn_norm___weight" : (None, 'fsdp', ), #  torch.float32 (n, 4096,)
  "norm.weight" : ('fsdp', ), #  torch.float32 (4096,)
  "output.weight" : ('fsdp', ), #  torch.float32 (vocab_size, 4096)
}
```

The above are different sharding schemes. Because we are using gSPMD, we need some
mechanism of sharding the weights. Because we don't (can't) modify the model code
itself, we can just use a dictionary of names to keep that information


```python
class Trainer:

    def __init__(self, mesh):
        self.mesh = mesh
        self.x_sharding = jax.sharding.NamedSharding(self.mesh, P('fsdp'))
        self.replicated = jax.sharding.NamedSharding(self.mesh, P())

    def fit(self, model, loss_fn, data_loader):
        xla_env = torchax.default_env()
        jax.config.update('jax_enable_x64', False)
        xla_env._mesh = self.mesh
        xla_env.use_flash_attention = True

        jittable_mod = JittableModule(model)

        # split the params to the n devices

        def model_fn(weights, buffers, args):
            return jittable_mod.functional_call('forward', weights, buffers, args)


        jax_optimizer = optax.sgd(0.01)
        opt_state = torch_view(jax_optimizer.init(jax_view(jittable_mod.params)))

        train_step = torchax.train.make_train_step(
            model_fn, loss_fn, jax_optimizer,
            remat_policy=jax.checkpoint_policies.offload_dot_with_no_batch_dims('device', 'pinned_host'),
            mark_fsdp_sharding_axis='fsdp')

        print('Begining training')
        s = time.perf_counter()
        jax.profiler.start_trace('/tmp/tensorboard')
        print('start training')
        min_loop_time = 10000
        for i, item in enumerate(data_loader):
            inputs, labels = item
            # Move them to jax device
            inputs = inputs.to('jax')
            labels = labels.to('jax')

            # Shard them on batch dim for fsdp
            inputs.apply_jax_(sharded_device_put, self.x_sharding)
            labels.apply_jax_(sharded_device_put, self.x_sharding)

            if i == 0:
                train_step = helper.compile_step_func(
                    train_step,
                    jittable_mod.params, jittable_mod.buffers, opt_state, inputs, labels,
                    self.mesh
                )

            print('INPUT shape', inputs.shape)
            step_start = time.perf_counter()
            loss, jittable_mod.params, opt_state = train_step(
                jittable_mod.params, jittable_mod.buffers, opt_state, inputs, labels)
            # wait for iteration to finish to measure time
            torchax.interop.call_jax(jax.block_until_ready, (loss, jittable_mod.params))
            step_end = time.perf_counter()
            print(i, 'loss', loss, 'step latency: ', step_end - step_start)
            loop_time = step_end - step_start
            min_loop_time = min(min_loop_time, loop_time)
            print('======')
            if i >= 3:
                break
        jax.profiler.stop_trace()
        return min_loop_time
```

The trainer class is the training loop.
Few things to note:

1. The training loop is something that calls a `train_step` repeatedly.
   The `train_step` is function that maps (weights, buffer, optimizer_state, inputs, labels)
   to (loss, updated weight, updated optimizer state). Returning loss is not needed
   only there for printing out. The buffer argument is the non-trainable paramters, in
   our case, it holds the `freqs_cis` variable

   The `train_step` is roughly equivalent to the follwoing:

  ```python
  def train_step(weights, buffer, optimizer_state, inputs, label):
    optimizer = recreate optimizer from optimizer_state
    state_dict = weights + buffer
    result = torch.func.functional_call(model, state_dict, inputs)
    loss = loss_fn(result, label)
    loss.backward()
    optimizer.step()
    return loss, model.paramters(), optimizer.state_dict()
  ```

2. Here we are using a fake dataloader.

3. We are calling `jax.block_until_ready` to measure iteration time, this is not needed
   for real training jobs

4. We use `jax.profiler` to capture profiles. Tools listed in here: https://jax.readthedocs.io/en/latest/profiling.html
   all works out of the box.

5. `interop.call_jax` API is used whenever we need something from Jax. Those API can be
   wrapped and have the "jaxiness" hidden. However, I don't think we need to do such hidding.

6. Precompile: call to `helpers.compile_step_func`. This is not needed. If not used, then
   it will compile on the first invokation. However, triggering compilation manually
   allows to print some stats (such as GBs accessed), also will error if the input shape
   / layout / sharding changed in the future iterations. For example I got the below while developing:
  ```
  ValueError: Received incompatible devices for jitted computation. Got argument args[0]['layers.params.attention___wk___weight'] of <unnamed wrapped function> with shape bfloat16[32,1024,4096] and device ids [0, 2, 4, 6, 1, 3, 5, 7] on platform TPU and explicit output sharding with device ids [0] on platform TPU
  ```
  this tells me the sharding I specified was wrong and I would go back and fix.




```python
def _process_sharding_name(name):
    """Replace integers in param name with *.

  Presumably all layers should have the same sharding.
  """

    def is_integer(t):
        try:
            int(t)
            return True
        # pylint: disable-next=all
        except:  # noqa: E722
            return False

    tokens = name.split(".")
    for i, t in enumerate(tokens):
        if is_integer(t):
            tokens[i] = "*"
    return ".".join(tokens)
```
This is a helper to process names in sharding map


```python
def create_sharded_weights(model, mesh, sharding_map):
    res = {}
    env = torchax.default_env()
    for name, weight_meta in model.state_dict().items():
        sharding_spec = sharding_map.get(_process_sharding_name(name))
        if sharding_spec is None:
            print('Skipping weight:', name)
            continue
        sharding = NamedSharding(mesh, P(*sharding_spec))
        with jax.default_device(jax.devices('cpu')[0]):
            weight_torch = torch.randn(
              weight_meta.shape,
              dtype=weight_meta.dtype)
            weight_jax = torchax.default_env().to_xla(weight_torch).jax()
        #print(name, weight.shape, weight.dtype)
        res[name] = env.j2t_iso(jax.make_array_from_callback(
          weight_jax.shape, sharding, lambda a: weight_jax[a]
        ))
    return res
```
The strategy of not OOMing the host on larger scale training:
allocate the model on meta device, then re-initialize weights one by one,
shard the weight immediately after creation.


```python
def fake_dataloader(size, seqlen, batch_size):
  for _ in range(size):
    x = torch.randint(0, 32000, (batch_size, seqlen), device='cpu')
    yield x, (x + 1) % 32000
```

Fake dataloader, just create random ints of desired shape.


Then the below is the `main` function. I will split it into pieces for better commenting

```python
def main(
    model_type='8B',
    batch_size=8,
    seqlen=2048,
    override_num_layers=-1,
    use_scan = True,
    tp_parallelism=1,
):
    torchax.enable_globally()
    torchax.enable_performance_mode()
    #logging.getLogger("jax").setLevel(logging.DEBUG)
    print(f"Running with parameters {locals()}")

    fsdp = num_global_devices // tp_parallelism
    mesh = jax.make_mesh((fsdp, tp_parallelism), ('fsdp', 'tp'))
```
Above, the config is set to run either fsdp only or also with tensor parallelism.
If using tp (i.e. passing `tp_parallelism > 1`) then the global devices will be
split into fsdp x tp 2D array. Tensors will be sharded on those 2 axis

```python
    if use_scan:
        # using scan the individial weights will have shape (num_layers, w, h)
        sharding_map = sharding_map_scan_fsdp
    else:
        sharding_map = sharding_map_original
```
Scan is implemented as the `TransformerWithScan` below.

```python
    env = torchax.default_env()
    env.config.use_tpu_flash_attention = True
    env.config.shmap_flash_attention = True
    env._mesh = mesh  # this is the mesh used by flash attention pallas kernel
```
this bit tells TX to use flash_attention implemented in pallas. Because pallas is
single device by default, we apply `jax.shard_map` with a mesh.

```python
    args = llama3_configs[model_type]
    # Note: torchtitan's upstream config did not specify this value
    args.vocab_size = 128256
    args.max_seq_len = seqlen
    if override_num_layers > 0:
        args.n_layers = override_num_layers

    # Note: because a single device don't have enough HBM memory
    # nor enough CPU memory to hold the parameters. We instantiate
    # the model on meta then manually initialize then shard each param
    torch.set_default_dtype(torch.bfloat16)
    with torch.device('meta'):
        gpt = titan.Transformer(args)
```
Above, instantiate the model on meta device so no OOM.

```python
    with torch.device('cpu'):
        # need actual value for freqs_cis
        freqs_cis = gpt._precompute_freqs_cis()
```
Compute freqs_cis on CPU because we actually need its value.

```python
    if use_scan:
        checkpoint_policy=jax.checkpoint_policies.offload_dot_with_no_batch_dims('device', 'pinned_host')
        gpt = TransfomerWithScan(gpt, checkpoint_policy)

    state_dict = dict(gpt.state_dict())
    state_dict.pop('freqs_cis') # dont shard freqs_cis
    state_dict = create_sharded_weights(gpt, mesh, sharding_map)
    replicated = jax.sharding.NamedSharding(mesh, P())

    state_dict['freqs_cis'] = freqs_cis.to('jax').apply_jax(jax.device_put, replicated)
    gpt.load_state_dict(state_dict, assign=True)

    train_loader = fake_dataloader(10, seqlen, batch_size)
```
Put the sharded arrays inside of XLATensor back to the model with `load_state_dict`

```python
    # NOTE: overriding attention to capture mesh and sharding info
    partition = P('fsdp', 'tp', None, None)
    attention = functools.partial(
      splash_attn.tpu_splash_attention,
      mesh, partition, True)
    attention = jax.jit(attention)

    def custom_attention(
        query, key, value, attn_mask=None,
        dropout_p=0.0, is_causal=False,
        scale=None, enable_gqa=False):
                  #  batch, num of head, seq, dim
      jk, jq, jv = jax_view((query, key, value))
      res =  attention(jk, jq, jv, None)
      return torch_view(res)
    env.override_op_definition(torch.nn.functional.scaled_dot_product_attention, custom_attention)
```
Above, this bit is to showcase the "hackability": User can override the definition
of an torch op at runtime, and user's version will be invoked. Here I am using
jax pallas implementation of `splash_attention` i.e. sparse flash attention.
Note this can be done without modifying the model at all.
All ops that are `__torch_function__` capturable or `__torch_dispatch__` capturable are
eligible to be overriden.

```python
    def loss_fn(logits, y):
        num_tokens = logits.shape[-1]
        logits = logits.reshape(-1, num_tokens)
        y = y.reshape(-1)
        return torch.nn.functional.cross_entropy(
            logits, y)
```
Standard torch loss function. Needed reshape because `cross_entropy` only work with one
batch dim (not both batch and sequence)

```python
    with mesh:
        trainer = Trainer(mesh)
        return trainer.fit(
            gpt,
            loss_fn,
            train_loader
        )
```
Invoking the traininer.


```python
class TransfomerWithScan(torch.nn.Module):

    def __init__(self, old_transformer, checkpoint_policy):
        super().__init__()
        self.tok_embeddings = old_transformer.tok_embeddings
        self.norm = old_transformer.norm
        self.output = old_transformer.output
        self.layers = torchax.train.ScannedModule(list(old_transformer.layers.values()), checkpoint_policy)

        self.register_buffer('freqs_cis', old_transformer.freqs_cis)

    def forward(self, tokens: torch.Tensor):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        # for layer in self.layers.values():
        #     h = layer(h, self.freqs_cis)

        h = self.layers(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output
```
The goal of this class is to replace the for loop that iterate the layers with
a loop with scan. The use of scan is encaptured in `ScanedModule`. This class
is to override the `forward` to call `ScannedModule` instead of calling it in a loop.
