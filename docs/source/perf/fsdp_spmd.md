# Fully Sharded Data Parallel (FSDP) using SPMD

FSDP in PyTorch/XLA is a utility for
sharding Module parameters across data-parallel workers.

This differs from the [other](./fsdp_spmd.md) implementation of FSDP in PyTorch/XLA
in that this implementation uses SPMD.

Please review the SPMD user guide at {doc}`spmd_basic` before
proceeding. You can also find a minimum runnable example
[here](https://github.com/pytorch/xla/blob/master/examples/fsdp/train_decoder_only_fsdp_v2.py).

Example usage:

```python
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2

# Define the mesh following common SPMD practice
num_devices = xr.global_runtime_device_count()
mesh_shape = (num_devices, 1)
device_ids = np.array(range(num_devices))
# To be noted, the mesh must have an axis named 'fsdp', which the weights and activations will be sharded on.
mesh = xs.Mesh(device_ids, mesh_shape, ('fsdp', 'model'))

# Shard the input, and assume x is a 2D tensor.
x = xs.mark_sharding(x, mesh, ('fsdp', None))

# As normal FSDP, but an extra mesh is needed.
model = FSDPv2(my_module, mesh)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
output = model(x, y)
loss = output.sum()
loss.backward()
optim.step()
```

It is also possible to shard individual layers separately and have an
outer wrapper handle any leftover parameters. Here is an example to
autowrap each `DecoderLayer`.

```python
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Apply FSDP sharding on each DecoderLayer layer.
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        decoder_only_model.DecoderLayer
    },
)
model = FSDPv2(
    model, mesh=mesh, auto_wrap_policy=auto_wrap_policy)
```

## Sharding output

To ensure the XLA compiler correctly implements the FSDP algorithm, we
need to shard both weights and activations. This means sharding the
output of the forward method. Since the forward function output can
vary, we offer shard_output to shard activations in cases where your
module output doesn't fall into one of these categories: 1. A single
tensor 2. A tuple of tensors where the 0th element is the activation.

Example usage:

```python
def shard_output(output, mesh):
    xs.mark_sharding(output.logits, mesh, ('fsdp', None, None))

model = FSDPv2(my_module, mesh, shard_output)
```

## Gradient checkpointing

Currently, gradient checkpointing needs to be applied to the module
before the FSDP wrapper. Otherwise, recursively loop into children
modules will end up with infinite loop. We will fix this issue in the
future releases.

Example usage:

```python
from torch_xla.distributed.fsdp import checkpoint_module

model = FSDPv2(checkpoint_module(my_module), mesh)
```
