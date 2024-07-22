# Eager Mode + Compile API

In this doc we will go over how to use PyTorch/XLA’s new experimental `eager` mode with the `compile` API. The goal is to make PyTorch/XLA experience more aligned with the native PyTorch and make development process easier.


## Background
Currently PyTorch/XLA runs on the LazyTensor tracing mode by default. In the following code
```python
import torch
import torch_xla
import torchvision

device = torch_xla.device()
model = torchvision.models.resnet18().to(device)
input = torch.randn(64, 3, 224, 224).to(device)

# model tracing
res = model(input)

# model execution, same as `xm.mark_step`
torch_xla.sync()
```
The actual model compilation and device execution happens when `torch_xla.sync` is called. There are multiple drawback of this approach. 

1. Users are often confused about when the framework is tracing and when the framework is executing.
2. Non-core model code(data preprocessing for example) often generates some small pending execution that gets leaked into the main graph(step function) and causes recompilation. The recompilation of the whole graph is usually very expensive.
3. It is hard to debug when/why recompilation happens.

To mitigate above issues we want to introduce the new UX with eager and compile.

## Basic Usage
```python
import torch
import torch_xla
import torchvision

# Run ops eagerly by default
torch_xla.experimental.eager_mode(True)

device = torch_xla.device()
model = torchvision.models.resnet18().to(device)

# Mark the function to be compiled
compiled_model = torch_xla.experimental.compile(model)
input = torch.randn(64, 3, 224, 224).to(device)

# Compilation and execution happens right away.
res = compiled_model(input)
```
Note that

1. Currently user has to manually enable the eager mode by `torch_xla.experimental.eager_mode(True)`.
2. The region of the code that wants to be compiled should be wrapped by `torch_xla.experimental.compile`.

The implementation of the `torch_xla.experimental.compile` is actually pretty straight forward, it disable the eager mode when entering the target function and start tracing. It will call the `torch_xla.sync()` when target function returns and reenable the eager mode. You can expect the same perfomrance by using the `eager` + `compile` API compared to the existing `mark_step/sync` approach.


### Inference
```python
torch_xla.experimental.eager_mode(True)

compiled_model = torch.compile(model, backend="openxla")
```
It is recommened to use the `torch.compile` instead of `torch_xla.experimental.compile` for inference to reduce the tracing overhad. 

### Training
```python
torch_xla.experimental.eager_mode(True)

def step_fn(model, data, target, loss_fn, optimizer):
    optimizer.zero_grad()
    logits = model(data)
    loss = loss_fn(logits, target)
    loss.backward()
    optimizer.step()
    return loss

step_fn = torch_xla.experimental.compile(step_fn)
```
In training we asked user to refactor the `step_fn` out because it is usually better to compile the model's forward, backward and optimizer together. The long term goal is to also use `torch.compile` for training but right now we recommend user to use `torch_xla.experimental.compile`(for perfomrance reason).

## Benchmark

I run a 2 layer decoder only model training(it is pretty much just a llama2) with fake data on a single chip of v4-8 for 300 steps. Below is the number I observed.


<table>
  <tr>
   <td>
   </td>
   <td>token/s

  </tr>
  <tr>
   <td>Tracing mode(base line)
   </td>
   <td>147
   </td>
   </td>
  </tr>
  <tr>
   <td>Eager mode
   </td>
   <td>65
   </td>

   </td>
  </tr>
  <tr>
   <td>Eager + torch_xla compile
   </td>
   <td>147
   </td>
   </td>
  </tr>
</table>


Eager mode can achieve ~45% performance of the fully compiled model for the decoder only model. The trainer I used to test can be found [here](https://github.com/pytorch/xla/blob/master/examples/train_decoder_only_base.py) and [here](https://github.com/pytorch/xla/tree/master/examples/eager). Note that perfomrane of the eager mode is very model dependent. When I tried to run the resnet50, the eager mode perfomrance is ~1% of the compiled mode. We don't exepct user to use eager mode  to execute the main training loop. Eager mode is meant to be used to handle non-core part of the training/inference logic(Data preprocessing, random number generations etc) or debug.
