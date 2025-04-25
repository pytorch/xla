# Speed Up Tracing with `@assume_pure`

This document explains how to use `torch_xla.experimental.assume_pure` to
eliminate lazy tensor tracing overhead. See [this blog post][lazy-tensor] for a
primer on how lazy tensor tracing (operation recording) works.

## Background and motivation

PyTorch/XLA's lazy tensor tracing ensures correct execution by recording an
operation graph (lazy tensor IR) when running PyTorch operations. For complex
models, this tracing overhead can exceed the execution time of the graph,
leading to performance bottlenecks. When training a model, the layers in the
model must be re-traced on every training step. That's because there's no
guarantee that the layers will do the same thing in different training steps. As
an extreme example, a layer's `forward()` function may call `math.random()` and
decide what code to run based on a pseudo random number.

Re-tracing can introduce unnecessary overhead. In many cases, the layers in your
model will do exactly the same thing when given the same input tensor shapes. In
other words, given the same input, the function return the same output. Often,
the layers also will not will not perform side-effects such as saving the tensor
to a file or adding it to a global list. Such functions are called
"[pure functions][pure-function]".

Any PyTorch/XLA function decorated with `@assume_pure` will only be traced once
for each unique input tensor shape and dtype combination. PyTorch/XLA will cache
the traced computation instead of repeatedly tracing the same operations.

## How to use `@assume_pure`

### Using `@assume_pure` with a function

If you know your function is pure, decorate your function with `@assume_pure`:

```py
import torch
import torch_xla
from torch_xla.experimental.assume_pure import assume_pure

@assume_pure
def do_some_math(
    # You can pass any number of XLA tensors.
    a: torch.Tensor,
    b: torch.Tensor,

    # Non-tensor arguments are also supported, and passing different values will
    # trigger re-tracing and caching more computations.
    c: int,
):
    # Evaluate some pure expressions.
    return a @ b + c

# Simulate a training loop.
# Even if we run this function ten times, it will only be traced once.
for i in range(10):
    v = do_some_math(
        torch.tensor([1.0], device='xla'),
        torch.tensor([2.0], device='xla'),
        c=42,
    )
    print(v)
```

### Using `@assume_pure` with a `nn.Module`

If you have a pure `nn.Module` i.e. its `forward` behavior only depends on the
input arguments and the model parameters, we can use `torch.func.functional_call`
to convert the module into a pure function and pass that to `assume_pure`:

```python
import torch
import torch.nn as nn
from torch.func import functional_call
from torch_xla.experimental.assume_pure import assume_pure

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    def forward(self, x):
        return self.linear(x)

# Create module and move to XLA device
module = MyModule()
module = module.to('xla')

# Convert module's forward pass into a pure function
pure_forward = lambda params, buffers, x: functional_call(module, (params, buffers), (x,))

# Wrap the pure function with @assume_pure
cached_forward = assume_pure(pure_forward)

# Simulate a training loop
# Even if we run the model ten times, its forward function will only be traced once.
params = dict(module.named_parameters())
buffers = dict(module.named_buffers())
for i in range(10):
    x = torch.randn(5, 10, device='xla')
    y = cached_forward(params, buffers, x)
    print(y)
```

## Benchmarks

The unit tests contain a benchmark that traces an example 100 layer decoder-only
language model:

```sh
~/pytorch/xla
❯ TESTBRIDGE_TEST_ONLY=test_trace_transformer_with_spda_attention python3 test/test_assume_pure.py --benchmark_iterations 100
[...]
No `@assume_pure` time: 140.1342 ms
`@assume_pure` time: 24.1658 ms
```

The version with `@assume_pure` is much faster.

Importantly, the `@assume_pure` running time does not scale with increasing
complexity inside the model. That's because we only trace the model once, paying
a fixed up-front cost, and then later runs will reuse the cached XLA computation.

## Limitations

Currently, all operations in a function wrapped with `@assume_pure` must be
PyTorch upstream operations (e.g. `torch.einsum`, `torch.sin`, ...). More
PyTorch/XLA operations (e.g. `mark_sharding`) will be supported in the future.

<!-- xrefs -->

[lazy-tensor]: https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/
[pure-function]: https://en.wikipedia.org/wiki/Pure_function
