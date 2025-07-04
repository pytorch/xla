# Advanced Topics in PyTorch XLA

## Compilation, caching and execution

HLO is fed to the XLA compiler
for compilation and optimization. Compilation is then cached by PyTorch
XLA to be reused later if/when needed. The compilation of the graph is
done on the host (CPU), which is the machine that runs the Python code.
If there are multiple XLA devices, the host compiles the code for each
of the devices separately except when using SPMD (single-program,
multiple-data). For example, v4-8 has one host machine and [four
devices](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu_v4).
In this case the host compiles the code for each of the four devices
separately. In case of pod slices, when there are multiple hosts, each
host does the compilation for XLA devices it is attached to. If SPMD is
used, then the code is compiled only once (for given shapes and
computations) on each host for all the devices.

## Synchronous execution and blocking

The *synchronous* operations in Pytorch XLA, like printing, logging,
checkpointing or callbacks block tracing and result in slower execution.
In the case when an operation requires a specific value of an XLA
tensor, e.g. `print(xla_tensor_z)`, tracing is blocked until the value
of that tensor is available to the host. Note that only the part of the
graph responsible for computing that tensor value is executed. These
operations do not cut the IR graph, but they trigger host-device
communication through `TransferFromDevice`, which results in slower
performance.

## Barriers

A *barrier* is a special instruction that tells XLA to execute the IR
graph and materialize the tensors. This means that the PyTorch XLA
tensors will be evaluated, and the results will be available to the
host. The user-exposed barrier in Pytorch XLA is
[torch_xla.sync()](https://github.com/pytorch/xla/blob/bdceee54eca1269ee954f6cdd1868c584d0e88a4/torch_xla/core/xla_model.py#L808),
which breaks the IR graph and results in code execution on the XLA
devices. One of the key properties of `torch_xla.sync()` is that unlike
synchronous operations it does not block the further tracing while the
device is executing the graph. However, it does block access to the
values of the tensors that are being materialized.

The example in the LazyTensor guide illustrates what happens in a simple
case of adding two tensors. Now, suppose we have a for loop that adds
XLA tensors and uses the value later:

``` python
for x, y in tensors_on_device:
    z += x + y
```

Without a barrier, the Python tracing will result in a single graph that
wraps the addition of tensors `len(tensors_on_device)` times. This is
because the `for` loop is not captured by the tracing, so each iteration
of the loop will create a new subgraph corresponding to the computation
of `z += x+y` and add it to the graph. Here is an example when
`len(tensors_on_device)=3`.

![img](../_static/img/IRgraph_no_markstep.png)

However, introducing a barrier at the end of the loop will result in a
smaller graph that will be compiled once during the first pass inside
the `for` loop and will be reused for the next
`len(tensors_on_device)-1` iterations. The barrier will signal to the
tracing that the graph traced so far can be submitted for execution, and
if that graph has been seen before, a cached compiled program will be
reused.

``` python
for x, y in tensors_on_device:
    z += x + y
    torch_xla.sync()
```

In this case there will be a small graph that is used
`len(tensors_on_device)=3` times.

![img](../_static/img/IRgraph_markstep.png)

It is important to highlight that in PyTorch XLA Python code inside for
loops is traced and a new graph is constructed for each iteration if
there is a barrier at the end. This can be a significant performance
bottleneck.

## Graphs

The XLA graphs can be reused when the same computation happens on the
same shapes of tensors. If the shapes of the inputs or intermediate
tensors change, then the XLA compiler will recompile a new graph with
the new tensor shapes. This means that if you have dynamic shapes or if
your code does not reuse tensor graphs, running your model on XLA will
not be suitable for that use case. Padding the input into a fixed shape
can be an option to help avoid dynamic shapes. Otherwise, a significant
amount of time will be spent by the compiler on optimizing and fusing
operations which will not be used again.

The trade-off between graph size and compilation time is also important
to consider. If there is one large IR graph, the XLA compiler can spend
a lot of time on optimization and fusion of the ops. This can result in
a very long compilation time. However, the later execution may be much
faster, due to the optimizations that were performed during compilation.

Sometimes it is worth breaking the IR graph with `torch_xla.sync()`. As
explained above, this will result in a smaller graph that can be reused
later. However making graphs smaller can reduce optimizations that
otherwise could be done by the XLA compiler.

## Data Loading

Another important point to consider is
[MPDeviceLoader](https://github.com/pytorch/xla/blob/a1f822e2627a5639464273241821852677401026/torch_xla/distributed/parallel_loader.py#L186).
Once your code is running on an XLA device, consider wrapping the torch
dataloader with XLA `MPDeviceLoader` which preloads data to the device
to improve performance and includes `torch_xla.sync()` in it. The latter
automatically breaks the iterations over batches of data and sends them
for execution. Note, if you are not using MPDeviceLoader, you might need
to set `barrier=True` in the `optimizer_step()` to enable
`torch_xla.sync()` if running a training job or explicitly adding
`torch_xla.sync()`.

**Note:**

0 and 1 are magic numbers in XLA and treated as constants in the
HLO. So if there is a random number generator in the code that can
generate these values, the code will compile for each value
separately. This can be disabled with `XLA_NO_SPECIAL_SCALARS=1`
environment variable.
