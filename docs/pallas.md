# Custom Kernels via Pallas

With the rise of OpenAI [triton](https://openai.com/research/triton), custom kernels become more and more popular in the GPU community, for instance, the introduction of [FlashAttention](https://github.com/Dao-AILab/flash-attention) and [PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html). In order to provide the feature parity in the TPU world, Google has introduced [Pallas](http://go/jax-pallas) and [Mosaic](http://go/mosaic-tpu). For PyTorch/XLA to continue pushing the performance in TPU, we have to support custom kernels, and the best way is through Pallas and Mosaic. The design doc is [TBA]().

Let's assume you have a Pallas kernel defined as follow:
```python3
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

## Adopt the above kernel to be compatible with PyTorch/XLA

Example usage:
```python3
q = torch.randn(3, 2, 128, 4).to("xla")
k = torch.randn(3, 2, 128, 4).to("xla")
v = torch.randn(3, 2, 128, 4).to("xla")

# Adopts any Pallas kernel
from torch_xla.experimental.custom_kernel import make_kernel_from_pallas
pt_kernel = make_kernel_from_pallas(add_vectors, lambda x, y: [(x.shape, x.dtype)])
output = pt_kernel(q, k)
```
For simple kernels, the adoption is just as simple as one liner. For more complicated kernels, you can refer to our Flash Attention implementation for details.

## Use built-in kernels

Besides manually wrapping external Pallas kernels, there are built-in kernels where the adoptions are done by PyTorch/XLA already.

Example usage:
```python3
# Use built-in kernels
from torch_xla.experimental.custom_kernel import flash_attention
output = flash_attention(q, k, v)
```

You can just use it like any other torch.ops.

## HuggingFace Llama 3 Example
We have a fork of HF Llama 3 to demonstrate a potential integration [here](https://github.com/pytorch-tpu/transformers/tree/alanwaketan/flash_attention).

## Dependencies
The Pallas integration depends on JAX to function. However, not every JAX version is compatible with your installed PyTorch/XLA. To install the proper JAX:
```bash
pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
```
