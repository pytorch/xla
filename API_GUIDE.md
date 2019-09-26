# PyTorch/XLA API And Best Practice

## XLA Tensors

PyTorch/XLA adds a new device, similarly to CPU and GPU devices. The following snippet creates an XLA tensor filled with random values, then prints the device and the contents of the tensor:

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

x = torch.randn(4, 2, device=xm.xla_device())
print(x.device)
print(x)
```

The XLA device is not a physical device but instead stands in for either a Cloud TPU or CPU. The underlying storage for XLA tensors is a contiguous buffer in device memory and the code in the model shouldn't assume any stride.

XLA Tensor doesn't support converting single tensor to half precision using `tensor.half()`. Instead, environment variable `XLA_USE_BF16` is available, which converts **all** PyTorch float values to bfloat16 when sending them to the TPU device. The conversion is totally transparent to the user, and the XLA tensors will still retain a float dtype. Similarly, when the tensor is moved back to CPU, its type will be float.

The [XLA readme](https://github.com/pytorch/xla/blob/master/README.md) describes all the options available to run on TPU or CPU.

## Running a model

There are different ways to run a model using the PyTorch/XLA framework.

### Native PyTorch API

The simplest (but not good performing) one is to just run on one core and send the input tensors to the XLA devices manually:

```python
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

The above is only running on one TPU core though, and the time spent to send data to device is serial/inline with the TPU computation.
For simple experiments, or for inference tasks which are not latency-sensitive it might be still OK, but the following methods allow for better scalability.

Note the `xm.optimizer_step(optimizer, barrier=True)` line which replaces the usual
`optimizer.step()`. This is required because of the way XLA tensors work:
operations are not executed immediately, but rather added to a graph of pending
operations which is only executed when its results are required. Using
`xm.optimizer_step(optimizer, barrier=True)` acts as an execution barrier which forces the
evaluation of the graph accumulated for a single step. Without this barrier, the
graph would only be evaluated when evaluating the accuracy of the model, which
is only done at the end of an epoch, for this example. Even for small models,
the accumulated graph would be too big to evaluate at the end of an entire
epoch.

### MultiCore

There are two ways to drive multiple TPU cores using PyTorch/XLA. One is using the `torch.multiprocessing` module (which internally spawns multiple processes), and the other is using Python threading.
The multiprocessing method should allow better performance as it gets around the Python GIL serialization, especially with model code which has a heavy Python side processing.
Note that in the MultiCore setting, a barrier is included inside the data
iterators, so there are no explicit `barrier=True` in the examples below.

#### MultiCore - MultiProcessing

Code for multiprocessing looks like:

```python
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(index):
  device = xm.xla_device()
  para_loader = dp.ParallelLoader(train_loader, [device])

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

One thing to remember about the multiprocessing approach is that `torch.multiprocessing` uses the `spawn` method of Python multiprocessing, which spawns completely new processes (contrary to forking) and send pickled data over pipes.
The only data which is pickled is that data passed to the target function of the `xmp.spawn()` API (the `args` argument), so if the parent process changes the global state before calling `xmp.spawn()`, such data won't be reflected into the child processes (unless passed into `args`).

Check the [full example](https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist.py) showing how to train MNIST on TPU using multiprocesing.

#### MultiCore - MultiThreading

To run a model using the Python threading support (embedded within the `torch_xla.distributed.data_parallel.DataParallel` interface), use the following API:

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

The same multi-core API can be used to run on a single core as well by setting the device_ids argument to the selected core. Passing `[]` as `device_ids` causes the model to run using the PyTorch native CPU support.

Check the [full example](https://github.com/pytorch/xla/blob/master/test/test_train_mnist.py) showing how to train MNIST on TPU using `torch_xla.distributed.data_parallel.DataParallel` (Python threading).

## Performance And Debugging

Model is still running slow after many iterations? Check out [troubleshooting guide](TROUBLESHOOTING.md) for tips about how to debug them!
