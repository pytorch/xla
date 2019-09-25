# Performance guideline for running on TPUs

## XLA tensors

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

## Performance caveats

PyTorch/XLA behaves semantically like regular PyTorch and XLA tensors, implementing the full tensor interface. However, constraints in XLA and hardware, and the lazy evaluation model mean some patterns must be avoided:

1.  Tensor shapes should be the same between iterations, or a low number of shape variations should be used. PyTorch/XLA automatically recompiles the graph every time new shapes are encountered. This means that, if the shapes don’t stabilize during training, more time will be spent compiling than running the model. Pad tensors to fixed sizes when possible. Direct or indirect uses of `nonzero` introduce dynamic shapes; for example, masked indexing `base[index]` where `index` is a mask tensor.
1.  Certain operations don’t have native translations to XLA and therefore require transfer to the CPU memory, evaluation on CPU, and transfer of the result back to the XLA device. This is automatically handled by PyTorch/XLA, but doing too many such operations during the training step can lead to significant slowdowns. The `item()` operation is one such example and it is used in [clip_grad_norm_](https://github.com/pytorch/pytorch/blob/de19eeee99a2a282fc441f637b23d8e50c75ecd1/torch/nn/utils/clip_grad.py#L33). Below is an alternative implementation which avoids the need for `item()`:

    ```python
    ...
    else:
      device = parameters[0].device
      total_norm = torch.zeros([], device=device if parameters else None)
      for p in parameters:
        param_norm = p.grad.data.norm(norm_type) ** norm_type
        total_norm.add_(param_norm)
      total_norm = (total_norm ** (1. / norm_type))
    clip_coef = torch.tensor(max_norm, device=device) / (total_norm + 1e-6)
    for p in parameters:
      p.grad.data.mul_(torch.where(clip_coef < 1, clip_coef, torch.tensor(1., device=device)))
    ```


1. In order to avoid recompilations, not only shapes must be constant, but also computations accross XLA devices in all hosts. A special case of this is loops with a different number of iterations between steps. PyTorch/XLA automatically handles them, but they are seen as different execution graphs and require recompilations.

1. Iterators in `torch_xla.distributed.data_parallel` may drop the
last few batches in the input iterator, in order to do the same amount of work
on all XLA devices. In the extreme case where dataset is small, and there are
too few steps, this may result in a no-op epoch. Therefore, it is better to use
small batch sizes in those cases.

1. Even when it's known that a PyTorch tensor is a scalar, avoid using
   `tensor.item()`. Prefer instead keeping it as a tensor and the use of tensor
   operations on it, using control flow substitutes such as `torch.where`.
   Following the latter approach will likely result in those operations behind
   fully fused within an XLA graph, without the need of issuing separate TPU
   computations. This can dramatically improve performance of the model, up to
   an N factor, where N is the number of `tensor.item()` calls per step.

`print(torch_xla._XLAC._xla_metrics_report())` can be used to print metrics at the end of each step to collect information regarding the number of compilations and operators that are part of the model but don’t have native XLA implementations. The `XLA_METRICS_FILE=/PATH/TO/FILE` environment setting can also be used to export per step metrics to a file.

In this report, any counter that starts with `aten::`
indicates a context switch between the XLA device and CPU, which can be a
potential performance optimization area in the model code.

# Debugging

Sometimes bad things happen and a deeper look into the _PyTorch/TPU_ stack is necessary.
In order to do that, _PyTorch/TPU_ has a series of environment variables and function calls
which can help understading its internal behavior.

Note that the infromation in this section is subject to be removed in future releases of
the _PyTorch/TPU_ software, since many of them are peculiar to a given internal implementation
which might change.

The _PyTorch/TPU_ stack keeps a series of metrics and counters during its execution, and
the following API returns a string representation of them:

```Python
torch_xla._XLAC._xla_metrics_report()
```

Printing out that information can help during the debug phases and while reporting issues.

The information included within the metrics report include things like how many time we
issue _XLA_ compilations, how long they take, how many times we execute, for how long,
how many device data handles we create/destroy, etc...
These information is reported in terms of percentiles of the samples.
An example is:

```
Metric: CompileTime
  TotalSamples: 202
  Counter: 06m09s401ms746.001us
  ValueRate: 778ms572.062us / second
  Rate: 0.425201 / second
  Percentiles: 1%=001ms32.778us; 5%=001ms61.283us; 10%=001ms79.236us; 20%=001ms110.973us; 50%=001ms228.773us; 80%=001ms339.183us; 90%=001ms434.305us; 95%=002ms921.063us; 99%=21s102ms853.173us
```

The _PyTorch/TPU_ stack also has counters, which are named integer variables tracks
internal software status.
Example:

```
Counter: CachedSyncTensors
  Value: 395
```

Counters are also useful to understand which operations the _PyTorch/TPU_ stack is routing
back to the CPU engine of _PyTorch_.
Things which looks like a _C++_ namespace are part of this category:

```
Counter: aten::nonzero
  Value: 33
```

There are also a number of environment variables which control the behavior of the _PyTorch/TPU_
software stack.
Setting such variables will cause different degrees of performance degradation, so they should
only be enabled for debugging.

* ```XLA_IR_DEBUG```: Enables the _Python_ stack trace to be catpured where creating IR nodes,
  hence allowing to understand which _PyTorch_ operation was responsible of generating such IR.

* ```XLA_HLO_DEBUG```: Enables the _Python_ stack frame captured when _XLA_IR_DEBUG_ is active,
  to be propagated to the _XLA_ _HLO_ metadata.

* ```XLA_SAVE_TENSORS_FILE```: The path to a file which will be used to dump the IR graphs during
  execution. Note that the file can become really big if the option is left enabled and the
  _PyTorch_ program let run for long time. The graphs are appended to the file, so to have a clean
  sheet from run to run, the file should be explicitly removed.

* ```XLA_SAVE_TENSORS_FMT```: The format of the graphs stored within the _XLA_SAVE_TENSORS_FILE_
  file. Can be ```text``` (the default), ```dot``` (the _Graphviz_ format) or ```hlo```.

* ```XLA_METRICS_FILE```: If set, the path to a local file where the internal metrics will be
  saved at every step. Metrics will be appended to the file, if already existing.

* ```GET_TENSORS_OPBYOP```: Enables pure _OpByOp_ dispatch. The _PyTorch/TPU_ software tries to
  fuse together many _PyTorch_ operations into a single computation graph, but sometimes, either
  for debugging, or in case the _PyTorch_ code have a very dynamic nature (in shapes or graph
  terms), it is better to force the execution in _OpByOp_ mode (every IR node is lowered into
  a separate _XLA_ computation, and chain-executed). This environment variable, if set to 1,
  enables _OpByOp_ during the "get tensors" operation (the operation used by _PyTorch/TPU_ to
  fetch intermediate values back from the _TPU_ device into _PyTorch_ CPU tensors).

* ```SYNC_TENSORS_OPBYOP```: The same as _GET_TENSORS_OPBYOP_ but for "sync tensors" operation
  (the operation used at the end of a step, to flush pending IR computations and materialize
  them into _TPU_ device data).

* ```XLA_SYNC_WAIT```: Forces the XLA tensor sync operation to wait for its completion, before
  moving to the next step.

* ```XLA_USE_BF16```: If set to 1, tranforms all the _PyTorch_ _Float_ values into _BiFloat16_
  when sending to the _TPU_ device.

* ```XLA_USE_32BIT_LONG```: If set to 1, maps _PyTorch_ _Long_ types to _XLA_ 32bit type.
  On the versions of the TPU HW at the time of writing, 64bit integer computations are
  expensive, so setting this flag might help. It should be verified by the user that truncating
  to 32bit values is a valid operation according to the use of _PyTorch_ _Long_ values in it.


