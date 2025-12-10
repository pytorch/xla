# PyTorch/XLA Compile API and it's interaction with Eager mode.

## Overview

PyTorch/XLA integrates PyTorch with the XLA compiler to optimize deep learning
workloads across various hardware accelerators. Currently PyTorch/XLA uses the
LazyTensor tracing mode by default where operations are recorded into a
computation graph for deferred compilation and execution (triggered by
`torch_xla.sync()`), as shown in the following code:

```python
import torch
import torch_xla
import torchvision

device = torch_xla.device()
model = torchvision.models.resnet18().to(device)
input = torch.randn(64, 3, 224, 224).to(device)

# model tracing
res = model(input)

# model execution
torch_xla.sync()
```

While this approach enables performance optimizations, it introduces significant
usability challenges.

## Challenges with LazyTensor Mode

- **Ambiguity**: Developers struggle to distinguish between tracing and
execution phases, complicating development and debugging.

- **Recompilation Overhead**: Whenever any part of the captured graph changes,
    `torch_xla.sync()` will recompile the whole graph. Changes in non-core
    operations (e.g., data preprocessing) thus trigger expensive recompilations.

- **Debugging Difficulty**: Identifying the cause of recompilations is
challenging due to the opaque nature of graph-building processes.

## Eager Mode and `torch_xla.compile`

To address these issues, PyTorch/XLA introduces an experimental eager mode
(enabled via `torch_xla.experimental.eager_mode(True)`) and the
`torch_xla.compile` API. This shift aligns PyTorch/XLA more closely with
native PyTorch, prioritizing developer experience while preserving
performance. Eager mode is likely to become the default in future releases.

- **Eager Mode**: Executes operations immediately, enhancing flexibility and
debugging but at a performance cost.

- **torch_xla.compile**: A decorator or wrapper that explicitly marks code
(e.g., a model or function) for XLA compilation within an eager context,
providing clear boundaries and immediate feedback.

**Note that `torch_xla.compile` is independently useful, even outside of eager
mode, providing benefits such as preventing dataloading operations from leaking
into the training loop graph by capturing them into a separate graph, and
catching accidental graph breaks when `full_graph=True` is specified.**

## How `torch_xla.compile` works

Let's have a look at a basic usage of `torch_xla.compile`:

```python
import torch
import torch_xla
import torchvision

# Run ops eagerly by default
torch_xla.experimental.eager_mode(True)

device = torch_xla.device()
model = torchvision.models.resnet18().to(device)

# Mark the function to be compiled
compiled_model = torch_xla.compile(model)
input = torch.randn(64, 3, 224, 224).to(device)

# Compilation and execution happens right away.
res = compiled_model(input)
```

where the implementation of `torch_xla.compile` can be summarized as follows:

1. **Disables Eager Mode**: Temporarily switches to tracing to build a
    computation graph.

2. **Traces Operations**: Records operations for XLA optimization.

3. **Compiles and Executes**: Triggers compilation and execution via an
    internal `torch_xla.sync()` call.

4. **Re-enables Eager Mode**: Resumes eager execution after compilation.

This "eager-to-lazy-to-eager" transition abstracts synchronization complexity,
balancing flexibility and performance.

## `torch_xla.compile` vs. `torch.compile`

The PyTorch ecosystem offers multiple compilation APIs, and understanding their
distinct roles, especially within PyTorch/XLA, is crucial for optimal
performance and development.

- `torch_xla.compile` is optimized for PyTorch/XLA training workflows. Designed
to work efficiently with the XLA backend for iterative training, it's the
recommended API for compiling training loops due to its observed performance
advantages. The best practice is to enclose the complete training step, e.g.
forward pass, loss calculation, backward pass, and optimizer step, within a
`step_fn` and then compiling this function.

``` python
torch_xla.experimental.eager_mode(True)

def step_fn(model, data, target, loss_fn, optimizer):
    optimizer.zero_grad()
    logits = model(data)
    loss = loss_fn(logits, target)
    loss.backward()
    optimizer.step()
    return loss

step_fn = torch_xla.compile(step_fn)
```

- `torch.compile` is PyTorch's general-purpose compilation API designed to
accelerate PyTorch models across various backends. For PyTorch/XLA, it uses the
`openxla` backend. We recommend `torch.compile` for PyTorch/XLA inference
because it lowers tracing overhead, leading to more efficient static inference
graphs. To use it with XLA, simply specify `backend="openxla"`.

``` python
torch_xla.experimental.eager_mode(True)
compiled_model = torch.compile(model, backend="openxla")
```

The long-term aim is for `torch.compile` to be the single compilation API for
both training and inference on XLA.

## Performance Benchmarks

To quantify the performance impact of torch_xla.compile and eager mode,
benchmarks were conducted under specific conditions. The benchmarks utilized a
2-layer decoder-only model, similar to Llama2, trained with fake data. The
training process spanned 300 steps on a single chip of a v4-8 TPU. The observed
performance, measured in tokens per second, clearly illustrates the impact of
different execution modes:

| Mode                        | token/s |
|-----------------------------|---------|
| Tracing mode (base line)    | 147     |
| Eager mode                  | 65      |
| Eager + torch_xla compile   | 147     |

Eager mode with `torch_xla.compile` matches the performance of traditional
LazyTensor tracing mode at `147` tokens/s, demonstrating a better user
experience without performance loss.

Pure eager mode's performance is model-dependent; it achieves ~45% of the fully
compiled model's performance for decoder-only models. However, for ResNet50,
pure eager mode was significantly slower (about 1% of compiled mode). For more
information, see [train_decoder_only_base.py](https://github.com/pytorch/xla/blob/master/examples/train_decoder_only_base.py)
and [eager example](https://github.com/pytorch/xla/tree/master/examples/eager).
This varying overhead means pure eager mode is not intended for main training or
inference loops. Its utility lies in non-core tasks like data preprocessing,
random number generation, custom utilities, or debugging, where immediate
execution is prioritized over throughput.
