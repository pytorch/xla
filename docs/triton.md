# Custom GPU Kernels via Triton

PyTorch/XLA now supports [Triton](https://openai.com/research/triton) kernels, enabling high-performance deep learning model execution on GPUs. Triton, a specialized language and compiler for GPU programming, empowers developers to write custom kernels that leverage the full potential of GPUs for various operations in deep learning models.

Given a Triton kernel defined as follows:
```python3
@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
  # Triton add kernel from https://github.com/openai/triton/blob/main/python/tutorials/01-vector-add.py#L28
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  tl.store(output_ptr + offsets, output, mask=mask)

```

We can run make this kernel a part of the PyTorch/XLA execution graph as follows:

```python3
import torch

import torch_xla.experimental.triton as xla_triton
import torch_xla

import triton
import triton.language as tl

size = 16
x = torch.arange(size, dtype=torch.int64).to("xla")
y = torch.arange(size, dtype=torch.int64).to("xla")
output = torch.empty_like(x)
block_size = 8
grid = (triton.cdiv(size, block_size),)

# triton_call takes the same arguments as the triton.jit function, in addition 
to the kernel itself and the grid that is used to execute the kernel.
All the tl.constexpr terms are passed as kwargs at the end.
payload = xla_triton.triton_call(
    x, y, output, size, kernel=add_kernel, grid=grid, BLOCK_SIZE=block_size)

# To make the triton kernel, a part of the PyTorch/XLA graph, we create a
# custom call node with the expected inputs, payload from triton_call,
# the output shapes and output dtypes. The payload already contains information
# regarding how the GPU buffers will be loaded when this node is executed.
output = torch_xla._XLAC._xla_gpu_custom_call([x, y], payload,
                                                [output.shape], [torch.int64])

```

For more complex kernels, you can also refer to the Triton Flash Attention kernel test in PyTorch/XLA.

## Dependencies
The Triton integration depends on the `triton` package to function. This code is tested with `triton==2.3.0`. To install:
```bash
pip install --no-deps triton==2.3.0