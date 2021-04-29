# PyTorch on XLA Devices

PyTorch runs on XLA devices, like TPUs, with the
[torch_xla package](https://github.com/pytorch/xla/). This document describes
how to run your models on these devices.

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
`xm.xla_device()` returns the current XLA device. This may be a CPU or TPU
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
linear = torch.nn.Linear(10, 20)
l_out = linear(l_in)
print(l_out)
# Input tensor is not an XLA tensor: torch.FloatTensor
```

will throw an error since the `torch.nn.Linear` module is on the CPU.

## Running Models on XLA Devices

Building a new PyTorch network or converting an existing one to run on XLA
devices requires only a few lines of XLA-specific code. The following snippets
highlight these lines when running on a single device and multiple devices with XLA
multi-processing.

### Running on a Single XLA Device

The following snippet shows a network training on a single XLA device:

```python
import torch_xla.core.xla_model as xm

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

  optimizer.step()
  xm.mark_step()
```

This snippet highlights how easy it is to switch your model to run on XLA. The
model definition, dataloader, optimizer and training loop can work on any device.
The only XLA-specific code is a couple lines that acquire the XLA device and
mark the step. Calling
`xm.mark_step()` at the end of each training
iteration causes XLA to execute its current graph and update the model's
parameters. See [XLA Tensor Deep Dive](#xla-tensor-deep-dive) for more on
how XLA creates graphs and runs operations.

### Running on Multiple XLA Devices with Multi-processing

PyTorch/XLA makes it easy to accelerate training by running on multiple XLA
devices. The following snippet shows how:

```python
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(index):
  device = xm.xla_device()
  mp_device_loader = pl.MpDeviceLoader(train_loader, device)

  model = MNIST().train().to(device)
  loss_fn = nn.NLLLoss()
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

  for data, target in mp_device_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    xm.optimizer_step(optimizer)

if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
```

There are three differences between this multi-device snippet and the previous
single device snippet:

- `xmp.spawn()` creates the processes that each run an XLA device.
- `MpDeviceLoader` loads the training data onto each device.
- `xm.optimizer_step(optimizer)` consolidates the gradients between cores and issues the XLA device step computation.

The model definition, optimizer definition and training loop remain the same.

> **NOTE:** It is important to note that, when using multi-processing, the user can start
retrieving and accessing XLA devices only from within the target function of
`xmp.spawn()` (or any function which has `xmp.spawn()` as parent in the call
stack).

See the
[full multiprocessing example](https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist.py)
for more on training a network on multiple XLA devices with multi-processing.

## XLA Tensor Deep Dive

Using XLA tensors and devices requires changing only a few lines of code. But
even though XLA tensors act a lot like CPU and CUDA tensors their internals are
different. This section describes what makes XLA tensors unique.

### XLA Tensors are Lazy

CPU and CUDA tensors launch operations immediately or <b>eagerly</b>. XLA tensors,
on the other hand, are <b>lazy</b>. They record operations in a graph until the
results are needed. Deferring execution like this lets XLA optimize it. A graph
of multiple separate operations might be fused into a single optimized
operation, for example.

Lazy execution is generally invisible to the caller. PyTorch/XLA automatically
constructs the graphs, sends them to XLA devices, and synchronizes when
copying data between an XLA device and the CPU. Inserting a barrier when
taking an optimizer step explicitly synchronizes the CPU and the XLA device. For
more information about our lazy tensor design, you can read [this paper](https://arxiv.org/pdf/2102.13267.pdf).

### XLA Tensors and bFloat16

PyTorch/XLA can use the
[bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
datatype when running on TPUs. In fact, PyTorch/XLA handles float types
(`torch.float` and `torch.double`) differently on TPUs. This behavior is
controlled by the `XLA_USE_BF16` environment variable:

- By default both `torch.float` and `torch.double` are
`torch.float` on TPUs.
- If `XLA_USE_BF16` is set, then `torch.float` and `torch.double` are both
`bfloat16` on TPUs.
- If a PyTorch tensor has `torch.bfloat16` data type, this will be directly
mapped to the TPU `bfloat16` (XLA `BF16` primitive type).

Developers should note that *XLA tensors on TPUs will always report their PyTorch datatype* regardless of
the actual datatype they're using. This conversion is automatic and opaque.
If an XLA tensor on a TPU is moved back to the CPU it will be converted
from its actual datatype to its PyTorch datatype. Depending on how your code operates, this conversion triggered by
the type of processing unit can be important.

### Memory Layout

The internal data representation of XLA tensors is opaque to the user. They
do not expose their storage and they always appear to be contiguous, unlike
CPU and CUDA tensors. This allows XLA to adjust a tensor's memory layout for
better performance.

### Moving XLA Tensors to and from the CPU

XLA tensors can be moved from the CPU to an XLA device and from an XLA device
to the CPU. If a view is moved then the data its viewing is also copied to the
other device and the view relationship is not preserved. Put another way,
once data is copied to another device it has no relationship with its
previous device or any tensors on it. Again, depending on how your code operates, 
appreciating and accommodating this transition can be important.

### Saving and Loading XLA Tensors

XLA tensors should be moved to the CPU before saving, as in the following
snippet:

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

This lets you put the loaded tensors on any available device, not just the one on which they were initialized.

Per the above note on moving XLA tensors to the CPU, care must be taken when
working with views. Instead of saving views it is recommended that you recreate
them after the tensors have been loaded and moved to their destination device(s).

A utility API is provided to save data by taking care of previously moving it
to CPU:

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

xm.save(model.state_dict(), path)
```

In case of multiple devices, the above API will only save the data for the master
device ordinal (0).

In case where memory is limited compared to the size of the model parameters, an
API is provided that reduces the memory footprint on the host:

```python
import torch_xla.utils.serialization as xser

xser.save(model.state_dict(), path)
```

This API streams XLA tensors to CPU one at a time, reducing the amount of host
memory used, but it requires a matching load API to restore:

```python
import torch_xla.utils.serialization as xser

state_dict = xser.load(path)
model.load_state_dict(state_dict)
```

Directly saving XLA tensors is possible but not recommended. XLA
tensors are always loaded back to the device they were saved from, and if
that device is unavailable the load will fail. PyTorch/XLA, like all of PyTorch,
is under active development and this behavior may change in the future.

## Further Reading

Additional documentation is available at the
[PyTorch/XLA repo](https://github.com/pytorch/xla/). More examples of running
networks on TPUs are available
[here](https://github.com/pytorch-tpu/examples).
