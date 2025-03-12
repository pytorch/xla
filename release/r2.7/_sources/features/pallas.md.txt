# Custom Kernels via Pallas

With the rise of OpenAI [Triton](https://openai.com/research/triton),
custom kernels become more and more popular in the GPU community, for
instance, the introduction of
[FlashAttention](https://github.com/Dao-AILab/flash-attention) and
[PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html). In order to
provide the feature parity in the TPU world, Google has introduced
[Pallas](https://jax.readthedocs.io/en/latest/pallas/index.html). For
PyTorch/XLA to continue pushing the performance in TPU, we have to
support custom kernels, and the best way is through Pallas.

Let's assume you have a Pallas kernel defined as follow:

``` python3
from torch_xla.experimental.custom_kernel import jax_import_guard
jax_import_guard()

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

def add_vectors_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y

@jax.jit
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(add_vectors_kernel,
                        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
                        )(x, y)
```

To be noted, it's very important to run `jax_import_guard()` before
importing any jax modules. Otherwise, the program will hang on TPU as
jax will lock the TPU and torch-xla cannot access it.

## Adopt the above kernel to be compatible with PyTorch/XLA

Example usage:

``` python3
q = torch.randn(3, 2, 128, 4).to("xla")
k = torch.randn(3, 2, 128, 4).to("xla")
v = torch.randn(3, 2, 128, 4).to("xla")

# Adopts any Pallas kernel
from torch_xla.experimental.custom_kernel import make_kernel_from_pallas
pt_kernel = make_kernel_from_pallas(add_vectors, lambda x, y: [(x.shape, x.dtype)])
output = pt_kernel(q, k)
```

For simple kernels, the adoption is just as simple as one liner. For
more complicated kernels, you can refer to our Flash Attention
implementation for details.

## Use built-in kernels

Besides manually wrapping external Pallas kernels, there are built-in
kernels where the adoptions are done by PyTorch/XLA already. These
built-in kernels can be used like any other torch.ops. The current
built-in kernels that are suppored are: - FlashAttention -PagedAttention

### FlashAttention

#### Example usage

``` python3
# Use built-in kernels
import torch_xla.experimental.custom_kernel
output = flash_attention(q, k, v)
```

#### Integration Example

We have an example of [FlashAttention integration
here](https://github.com/pytorch/xla/blob/master/examples/flash_attention/train_decoder_only_flash_attention.py)
in our training test script.

### PagedAttention

#### Example usage

``` python3
# Use built-in kernels
import torch_xla.experimental.custom_kernel
output = torch.ops.xla.paged_attention(
    query.squeeze(dim=1),
    key_cache,
    value_cache,
    context_lens,
    block_tables,
    pages_per_compute_block,
    megacore_mode=None,
)
```

#### Integration Example

The vLLM TPU integration utilizes [PagedAttention
here](https://github.com/vllm-project/vllm/blob/f5e1bf5d44877149eaabf9c04379a4e14a023145/vllm/attention/backends/pallas.py#L194)
for effective memory management with KV cache.

## Dependencies

The Pallas integration depends on JAX to function. However, not every
JAX version is compatible with your installed PyTorch/XLA. To install
the proper JAX:

``` bash
pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
```
