# Tracing Time vs. Execution Time in PyTorch/XLA

When working with PyTorch/XLA, it's essential to understand that operations on
XLA tensors are not typically executed immediately in the way they are with
standard PyTorch tensors on CPU or CUDA devices (which operate in "eager mode").
PyTorch/XLA employs a "lazy execution" model. This means that when you write
PyTorch code using XLA tensors, you are primarily defining or tracing a
computation graph. The actual execution of this graph on the XLA device (like a
TPU) is deferred until a specific trigger point.

This leads to two distinct types of "time" to consider:

1. Tracing Time: The time spent by PyTorch/XLA to record operations, build the
computation graph.
1. Execution Time: The actual time the XLA device spends compiling (if
necessary) and executing the computation graph.

## Tracing: Defining What to Compute

When you write PyTorch code using XLA tensors (e.g., tensors on a TPU),
PyTorch/XLA doesn't execute each operation on the device right away. It traces
these operations, adding them to an internal computation graph. If you measure
the duration of code that only performs XLA operations without an explicit
instruction to wait for the device, you are primarily measuring this tracing
time plus Python overhead.

### Illustrating Tracing-Dominant Measurement

Consider the following conceptual code:

```Python
# Assume 'a' and 'b' are XLA tensors
# start_time = time.perf_counter()

# This operation is recorded in PyTorch/XLA's graph
result = torch.matmul(a, b)

# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
```

The `elapsed_time` here would predominantly reflect how long it took PyTorch/XLA
to trace the matmul operation. The actual matrix multiplication on the XLA
device is not started.

## Execution: Device Computation

To accurately measure XLA device computation time:

1. Ensure the traced computational graph is sent to the device for execution.

1. Make sure your Python script pauses its timer until the device finishes processing.

### Triggering Execution and Ensuring Completion

Several mechanisms trigger graph execution and/or ensure completion:

1. `torch_xla.sync(wait=True)`: This is the most direct method for benchmarking.
It ensures all pending XLA operations are launched and, crucially, blocks the
Python script until the device finishes.

1. `Data Access/Transfer`: Operations like `tensor.cpu()`, `tensor.item()`, or
printing an XLA tensor require the actual data. To provide it, PyTorch/XLA must
execute the graph that produces the tensor and wait for its completion.

1. `torch_xla.core.xla_model.mark_step()`: Commonly used in loops, this signals
to XLA that a logical unit of work (e.g., a training step) is defined and can be
executed. It triggers execution but doesn't inherently block Python until device
completion.

1. `torch_xla.core.xla_model.optimizer_step(optimizer)`: In training, this
usually implies `mark_step` and ensures relevant computations are done.

1. `torch_xla.core.xla_model.unlazy(tensors)`: Blocks until specified tensors
are materialized.

### Measuring Full Execution Time

To accurately measure the time including device computation:

```Python
# Assume 'a' and 'b' are XLA tensors
# start_time = time.perf_counter()

result = torch.matmul(a, b)

# Explicitly wait for the XLA device to finish all pending operations
xm.sync(wait=True)

# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
```

Here, `elapsed_time` would include the tracing time, dispatch overhead, and the
actual XLA device computation time for the matmul.

## Case Study: Timing Loops with `mark_step`

A common scenario involves loops, such as in model training, where
`mark_step` is used. Consider this structure:

```Python
def run_model_step():
    #... XLA tensor operations...
    pass

# start_loop_time = time.perf_counter()
for step in range(num_steps):
  run_model_step()  # Operations are traced
  torch_xla.core.xla_model.mark_step()    # Graph for this step is submitted for execution
# end_loop_time = time.perf_counter()
# elapsed_loop_time = end_loop_time - start_loop_time
```

The `elapsed_loop_time` in this case primarily measures the cumulative host-side
time. This includes:

1. The time spent in `run_model_step()` for each iteration, which is largely
tracing XLA operations.

1. The time taken by `mark_step` in each iteration to process the graph for that
step and dispatch it to the XLA device for execution.

Crucially, `mark_step` typically allows the XLA device to execute
asynchronously. The Python loop can proceed to trace the next step while the
device is still working on the current or previous step's computations. Thus,
`elapsed_loop_time` does not guarantee inclusion of the full device execution
time for all `num_steps` if the device work lags behind the Python loop.

In order to Measure total loop time (including all device execution),
`torch_xla.sync(wait=True)` has to be added after the loop and before taking the
final timestamp.

```Python
# start_loop_time = time.perf_counter()
for step in range(num_steps):
  run_model_step()
  torch_xla.core.xla_model.mark_step()

xm.sync(wait=True)

end_loop_time = time.perf_counter()
elapsed_loop_time = end_loop_time2 - start_loop_time
```
