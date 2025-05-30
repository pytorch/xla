# Tracing Time vs. Execution Time in PyTorch/XLA

When working with PyTorch/XLA, it's essential to understand that operations on
XLA tensors are not typically executed immediately in the way they are with
standard PyTorch tensors on CPU or CUDA devices (which operate in "eager mode").
PyTorch/XLA employs a "lazy execution" model. This means that when you write
PyTorch code using XLA tensors, you are primarily defining or tracing a
computation graph. The compilation of the currently traced graph and it's
subsequent execution on the device are deferred until a specific trigger point.

This leads to two distinct types of "time" to consider:

1. Host-Side Time: The period during which your CPU (host) prepares the
computation. This includes:

   - **Tracing Time**: The period during which PyTorch/XLA records your
   operations and builds the computation graph.

   - **Compilation Time**: The time the host-side XLA compiler takes to
   transform the traced graph into optimized device code. This is most
   significant on the first execution of a new graph or if the graph changes.

2. Device Time: This is primarily the **Execution Time**, which is the period
   during which the XLA device (e.g., TPU) spends running the compiled code.

## Illustrating a Common Pitfall: Measuring Only Tracing Time

When you write PyTorch code using XLA tensors (e.g., tensors on a TPU),
PyTorch/XLA doesn't execute each operation on the device right away. It traces
these operations, adding them to an internal computation graph. **If you measure
the duration of code that only performs XLA operations without an explicit
instruction to wait for the device, you are primarily measuring this tracing
time plus Python overhead.**

Consider the following conceptual code:

```Python
# Assume 'a' and 'b' are XLA tensors
start_time = time.perf_counter()

# This operation is recorded in PyTorch/XLA's graph
result = torch.matmul(a, b)

# ❌❌❌ !!! INCORRECT PROFILING: compilation and execution are deferred !!! ❌❌❌
end_time = time.perf_counter()
elapsed_time = end_time - start_time
```

The `elapsed_time` here would predominantly reflect how long it took PyTorch/XLA
to trace the matmul operation. The actual matrix multiplication on the XLA
device, along with its compilation, is not started.

## Measuring End-to-End Performance

To correctly profile the performance of your code on the XLA device, you must
ensure that your timing captures host-side compilation and devide execution.
This involves:

1. Ensure the traced computational graph is compiled, if it's the first time
this graph is seen or if it is changed, and sent to the device for execution.

1. Make sure the Python script waits until the XLA device has completed all its
assigned computations before taking the final timestamp.

This is exemplified, using `torch_xla.sync(wait=True)`, in the following
conceptual code:

```Python
# Assume 'a' and 'b' are XLA tensors

# -- Warm-up Iteration begin ---

# The first execution of a new graph will include compilation time, as
# PyTorch/XLA translates the graph into optimized device code. To isolate the
# steady-state device execution time for consistent benchmarking, we perform a
# "warm-up" run.
_ = torch.matmul(a, b) # The result isn't needed, just triggering the op
torch_xla.sync(wait=True)

# -- Warm-up Iteration end ---

# ✅✅✅ CORRECT PROFILING
# Measure the steady-state execution time, which should exclude
# most of the initial compilation overhead.
start_time = time.perf_counter()

result = torch.matmul(a, b)

# Explicitly wait for the XLA device to finish.
torch_xla.sync(wait=True)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
```

### Triggering Execution and Ensuring Completion

Several mechanisms trigger graph execution and/or ensure completion:

1. `torch_xla.sync(wait=True)`: This is the most direct method for benchmarking.
It ensures all pending XLA operations are launched and, crucially, blocks the
Python script until the device finishes.

1. `Data Access/Transfer`: Operations like `tensor.cpu()`, `tensor.item()`, or
printing an XLA tensor require the actual data. To provide it, PyTorch/XLA must
execute the graph that produces the tensor and wait for its completion.

1. `torch_xla.core.xla_model.optimizer_step(optimizer)`: Reduces gradients,
applies optimizer updates, and conditionally triggers `torch_xla.sync` via its
barrier argument (default False, as data loaders often handle the sync).

1. `torch_xla.core.xla_model.unlazy(tensors)`: Blocks until specified tensors
are materialized.

## Case Study: Correctly Profiling Loops with `torch_xla.sync`

A common scenario involves loops, such as in model training, where
`torch_xla.sync` is used. Consider this structure:

```Python
def run_model():
    #... XLA tensor operations...
    pass

start_loop_time = time.perf_counter()
for step in range(num_steps):
  run_model()  # Operations are traced
  torch_xla.sync()    # Graph for this step is submitted for execution

# ❌❌❌ !!! INCORRECT PROFILING APPROACH FOR TOTAL TIME !!! ❌❌❌
end_loop_time = time.perf_counter()
elapsed_loop_time = end_loop_time - start_loop_time
```

The `elapsed_loop_time` in this case primarily measures the cumulative host-side
time. This includes:

1. The time spent in `run_model()` for each iteration (largly tracing).

1. The time taken by `torch_xla.sync` in each iteration to trigger the host-side
compilation (if the graph is new or changed) and dispatch the graph for that
step to the XLA device for execution.


Crucially, the graph submitted by `torch_xla.sync()` runs asynchronously: The
Python loop proceed to trace the next step while the device is still performing
its execution for the current or previous step. Thus, `elapsed_loop_time` does
not guarantee inclusion of the full device execution time for all `num_steps` if
the device work lags behind the Python loop.

In order to measure total loop time (including all device execution),
`torch_xla.sync(wait=True)` has to be added after the loop and before taking the
final timestamp.

```Python
start_loop_time = time.perf_counter()
for step in range(num_steps):
  run_model_step()
  torch_xla.sync()

# ✅✅✅ CORRECT PROFILING: Wait for ALL steps to complete on the device.
torch_xla.sync(wait=True)

end_loop_time = time.perf_counter()
elapsed_loop_time = end_loop_time - start_loop_time
```
