# PyTorch on XLA Devices

PyTorch runs on XLA devices, like Cloud TPUs, with the [torch_xla package](https://github.com/pytorch/xla/). This document describes how to run your models on these devices.

## Creating an XLA Tensor

PyTorch/XLA adds a new `xla` device type to PyTorch. This device type works just
like other PyTorch device types. For example, here's how to create and
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
`xm.xla_device()` returns the current XLA device. This may be a CPU or Cloud TPU
depending on your environment.

## XLA Tensors are PyTorch Tensors

PyTorch operations can be performed on XLA tensors just like CPU or CUDA tensors.

For example, XLA tensors can be added together:

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
same device. So code like

```python
l_in = torch.randn(10, device=xm.xla_device())
linear = torch.nn.Linear(10, 20).cpu()
l_out = linear(l_in)
print(l_out)
# Input tensor is not an XLA tensor: torch.FloatTensor
```

will throw an error since the torch.nn.Linear module is on the CPU.

## Running Models on XLA Devices

Building a new PyTorch network or converting an existing one to run on XLA devices requires only a few lines of XLA-specific code. The following snippets highlight these lines when running on a single device, multiple devices with XLA multiprocessing, or multiple threads with XLA multithreading.

### Running on a Single XLA Device

The following snippet shows a network training on one of several
device types:

```python
import torch_xla.core.xla_model as xm

device = 'cpu'
if ON_CUDA:
  device = 'cuda'
if ON_XLA:
  device = xm.xla_device()

model = MNIST().train().to(device)
loss_fn = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

for data, target in train_loader:
  optimizer.zero_grad()
  data = data.to(device)
  target = target.to(device)
  output = model(data)
  loss = loss_fn(output, target)
  loss.backward()

  if ON_XLA:
    xm.optimizer_step(optimizer, barrier=True)
  else:
    optimizer.step()

```

This snippet highlights how easy it is swap the device type your model runs
on. The model definition, dataloader, optimizer and training loop are common
to all device types. The only XLA-specific code is a couple lines that acquire
the XLA device and step the optimizer with a `barrier`. Calling
`xm.optimizer_step(optimizer, barrier=True)` at the end of each training
iteration causes XLA to execute its current graph and update the model's
parameters. See [XLA Tensor Deep Dive](#xla-tensor-deep-dive) for more on
how XLA creates graphs and runs operations.

### Running on Multiple XLA Devices with MultiProcessing

PyTorch/XLA makes it easy to accelerate training by running on multiple devices.
The following snippet shows how:

```python
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(index):
  device = xm.xla_device()
  para_loader = pl.ParallelLoader(train_loader, [device])

  model = MNIST().train().to(device)
  loss_fn = nn.NLLLoss()
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

  for data, target in para_loader.per_device_loader(device):
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    xm.optimizer_step(optimizer)

if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
```

There are three differences between this multidevice snippet and the previous
single device snippet:

- `xmp.spawn()` creates the processes that each run an XLA device.
- `ParallelLoader` loads the training data onto each device.
- `xm.optimizer_step(optimizer)` no longer needs a `barrier`. ParallelLoader
automatically creates an XLA barrier that evalutes the graph.

The model definition, optimizer definition and training loop remain the same.

See the [full multiprocessing example](https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist.py) for more on training a network on multiple XLA devices with multiprocessing.

### Running on Multiple XLA Devices with MultiThreading

Running on multiple devices using processes (see above) is preferred to using
threads. If, however, you want to use threads then PyTorch/XLA has a `DataParallel` interface. The following snippet shows the same network training with multiple threads:

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

The only differences between the multithreading and multiprocessing code are:

- Multiple devices are acquired in the same process with `xm.get_xla_supported_devices()`.
- The model is wrapped in `dp.DataParallel` and passed both the training loop
and dataloader.

See the [full multithreading example](https://github.com/pytorch/xla/blob/master/test/test_train_mnist.py) for more on training a network on multiple XLA
devices with multithreading.

## XLA Tensor Deep Dive

Using XLA tensors and devices requires changing only a few lines of code. But
even though XLA tensors act a lot like CPU and CUDA tensors their internals are
different. This section describes what makes XLA tensors unique.

### XLA Tensors are Lazy

CPU and CUDA tensors launch operations immediately or `eagerly`. XLA tensors,
on the other hand, are `lazy`. They record operations in a graph until the
results are needed. Deferring execution like this lets XLA optimize it. A graph
of multiple separate operations might be fused into a single optimized operation, for example.

Lazy execution is generally invisible to the caller. PyTorxh/XLA automatically
constructs the graphs, sends them to XLA devices, and synchronizes when
copying data between an XLA device and the CPU. Inserting a `barrier` when
taking an optimizer step also synchronizes the CPU and the XLA device.

### XLA Tensors and bFloat16

When Cloud TPUs are used as XLA devices then:

- Floating point (float and double) XLA tensors use the same datatype
- If the `XLA_USE_BF16` environment variable is set this datatype is `bfloat16`.
If not set it's fp32.

XLA tensors on Cloud TPUs will report their PyTorch datatypes regardless of
the actual datatype they're using. If moved to the CPU they will be converted
from their actual datatype to their PyTorch datatype. For example, if
`XLA_USE_BF16=1` then a XLA tensor with dtype `torch.double` on a Cloud TPU is
actually using `bfloat16`, and if that tensor is moved to the CPU the resulting
tensor will have dtype `torch.double`.

### Saving and Loading XLA Tensors

XLA tensors should be moved to the CPU before saving, as in the following snippet:

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device()

t0 = torch.randn(2, 2, device=device)
t1 = torch.randn(2, 2, device=device)

tensors = (t0.cpu(), t1.cpu())

torch.save(tensors, 'tensors.pt')

tensors = torch.load('tensors.pt')

t0 = tensors[0].to(device)
t1 = tensors[1].to(device)
```

The loaded tensors can be put on any available device.

While XLA tensors can be saved directly it is not recommended to do so. XLA
tensors are always loaded back to the device they are saved from, and if
that device is unavailable the load will fail. PyTorchXLA, like all of PyTorch,
is under active development and this behavior may change in the future.

## Further Reading

Additional documentation is available at the [PyTorch/XLA repo](https://github.com/pytorch/xla/). More examples of running networks on Cloud TPUs are available [here](https://github.com/pytorch-tpu/examples).
