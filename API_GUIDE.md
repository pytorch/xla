# PyTorch on TPUs

PyTorch runs on TPUs with the [torch_xla package](https://github.com/pytorch/xla/). This document describes how to run your models on `xla` devices.

## Creating an XLA Tensor

PyTorch/XLA adds a new `xla` device type to PyTorch. This device type works like
just other PyTorch device types. For example, the here's how to create and
print an XLA tensor:

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

t = torch.randn(2, 2, device=xm.xla_device())
print(t.device)
print(t)
```

This code should look familiar. PyTorch/XLA uses the same interface as regular
PyTorch with a few additions. Importing `torch_xla` initializes PyTorch/XLA, and
`xm.xla_device()` returns the current XLA device. This may be a CPU or Cloud TPU depending on your environment.

## Working with XLA Tensors

Operations can be performed on XLA tensors just like CPU or CUDA tensors. XLA
tensors can be added together:

```python
t0 = torch.randn(2, 2, device=xm.xla_device())
t1 = torch.randn(2, 2, device=xm.xla_device())
print(t0 + t1)
```

Or matrix multiplied:

```python
print(t0.mm(t1))
```

Or used with neural network modules:

```python
l_in = torch.randn(10, device=xm.xla_device())
linear = torch.nn.Linear(10, 20).to(xm.xla_device())
l_out = linear(l_in)
print(l_out)
```

Like other device types, XLA tensors only work with other XLA tensors on the
same device. So

```python
l_in = torch.randn(10, device=xm.xla_device())
linear = torch.nn.Linear(10, 20).cpu()
l_out = linear(l_in)
print(l_out)
# Input tensor is not an XLA tensor: torch.FloatTensor
```

will throw an error since the torch.nn.Linear module is on the CPU.

Even though XLA tensors are lot like CPU and CUDA tensors, there are a few
things that make them unique.

## XLA Tensors are Lazy

CPU and CUDA tensors launch operations immediately or `eagerly`. XLA tensors,
on the other hand, are `lazy`. They record the operations performed on them
into a graph until the results are needed. Deferring execution like this lets XLA optimize it. A graph of multiple separate operations might be fused into a single optimized operation, for example.

XLA tensors executing lazily, like CUDA tensors executing asynchronously, is
generally invisible to the caller. PyTorch/XLA automatically constructs the graphs,
sends them to XLA devices, and performs the needed synchronization when
copying data between an XLA device and the CPU.

## XLA Tensors on Cloud TPUs can use bFloat16

PyTorch/XLA supports the `bfloat16` data type on Cloud TPUs. When the
`XLA_USE_BF16` environment variable is set all PyTorch float scalars and tensors
will automatically be converted to bfloat16 on Cloud TPUs. They will be converted
back to PyTorch floats if moved to the CPU.

Cloud TPUs also treat the `torch.double` datatype as `torch.float`, so when
`XLA_USE_BF16` is set then `torch.double` actually means `torch.float` and
`torch.float` actually means `torch.bfloat16` when tensors are on Cloud TPUs.

## Running Models on XLA Devices

### Getting Started

The following snippet shows a network running on a single XLA device:

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device()
model = MNIST()
loss_fn = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

model.train()
for data, target in train_loader:
  optimizer.zero_grad()
  data = data.to(device)
  target = target.to(device)
  output = model(data)
  loss = loss_fn(output, target)
  loss.backward()
  xm.optimizer_step(optimizer, barrier=True)
```

Like in the previous snippets, this snippet acquires the XLA device after
importing `torch_xla` with `xm.xla_device()`. The network
is constructed as usual and its inputs are sent to the XLA device after being
loaded on the CPU. At the end of each training iteration
`xm.optimizer_step(optimizer, barrier=True)` is called. This replaces the
usual Pytorch `optimizer.step()` and causes XLA to execute its current graph
and update the model's parameters.

### Running on Multiple Devices with MultiProcessing

The snippet shown above only ran the network on a single device. PyTorch/XLA
makes it easy to accelerate training by running on multiple devices. The
following snippet shows how:

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(index):
  device = xm.xla_device()
  para_loader = pl.ParallelLoader(train_loader, [device])

  model = MNIST()
  loss_fn = nn.NLLLoss()
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

  model.train()
  for data, target in para_loader.per_device_loader(device):
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    xm.optimizer_step(optimizer)

if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
```

There are only three differences between this multiprocessing snippet and the
previous one. First, `xmp.spawn()` is used to create the processes that will
run on other XLA devices. Second, a `ParallelLoader` is used to load the
training data onto each device. Third, there's no need to call
`xm.optimizer_step(optimizer)` with `barrier=True` since ParallelLoader
automatically creates an XLA barrier that forces evaluation of the graph.

See the [full multiprocessing example](https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist.py) of how to train MNIST on multiple Cloud TPUs with multiprocessing.

### Running on Multiple Devices with MultiThreading

Running on multiple devices using processes (see above) is preferred to using
threads. If, however, you prefer to use threads then PyTorch/XLA provides
that support with its `DataParallel` interface. The following snippet shows
the same network training with multiple threads:

```python
import torch_xla.core.xla_model as xm
import torch_xla.distributed.data_parallel as dp

devices = xm.get_xla_supported_devices()
model_parallel = dp.DataParallel(MNIST, device_ids=devices)

def train_loop_fn(model, loader, device, context):
  loss_fn = nn.NLLLoss()
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

  model.train()
  for _, (data, target) in loader:
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    xm.optimizer_step(optimizer)

for epoch in range(1, num_epochs + 1):
  model_parallel(train_loop_fn, train_loader)
```

This multithreaded snippet differs from the multiprocessing example by first
initializing itself with `DataParallel` and creating threads with
`model_parallel(train_loop_fn, train_loader)` instead of creating
processes with `xmp.spawn()`.

See the [full multithreading example](https://github.com/pytorch/xla/blob/master/test/test_train_mnist.py) of how to train MNIST on multiple Cloud TPUs with
multithreading.

## XLA Tensor Internals

As the above snippets and examples show, using an XLA device or even multiple
XLA devices requires changing only a few lines of code. The internals of an XLA
tensor, however, are different from those of CPU or CUDA tensors. In particular,
XLA tensors have no storage and always appear to be contiguous.

These differences mean that XLA tensors behave slightly differently when being
saved, loaded, and copied. PyTorch/XLA, like all of PyTorch, is actively
developed and these behaviors may change in the future.

### Saving and Loading XLA Tensors

XLA tensors can be saved and loaded like CPU and CUDA tensors, as in the
following snippet:

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

t0 = torch.randn(2, 2, device=xm.xla_device())
t1 = torch.randn(2, 2, device=xm.xla_device())

tensors = (t0, t1)

torch.save(tensors, 'tensors.pt')

tensors = torch.load('tensors.pt')

print(tensors)
```

An XLA tensor and a view of that tensor cannot be saved together, however.
The following snippet will throw an error:

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

t = torch.randn(2, 2, device=xm.xla_device())
t_view = t[0:2]

tensors = (t, t_view)

torch.save(tensors, 'tensors.pt')
# RuntimeError: Tensor ID 2 is sharing a view with tensor ID 1
```

To save an XLA tensor with views you can move it to the CPU before saving it.
Converting XLA tensors to CPU tensors before saving also lets them
be loaded as CPU tensors and placed on any available device. When XLA tensors
are are loaded using `torch.load()` they are placed back on the XLA device
they were saved from. If these devices are unavailable the load will fail.
Setting the `map_location` in `torch.load` is a no-op for XLA tensors.

### Copying XLA Tensors

Copying an XLA tensor using Python's `copy.copy()` returns a deep copy instead of shallow copy. Use `tensor.view()` instead if you want a shallow copy.
