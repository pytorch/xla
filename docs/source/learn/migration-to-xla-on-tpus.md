# Migrating from PyTorch on GPUs to PyTorch/XLA on TPUs

This guide is for developers already familiar with training PyTorch models on GPUs who want to transition to using Google Cloud TPUs with PyTorch/XLA. While PyTorch/XLA aims for a seamless experience, there are key differences in hardware architecture and execution models that necessitate some adjustments to your code and workflow for optimal performance.

## Key Differences: GPUs vs. TPUs with PyTorch/XLA

Understanding these differences will help you adapt your code effectively:

**Execution Model:**

* **GPUs (CUDA)**: Typically use an *eager execution* model where operations are dispatched and executed immediately.
* **TPUs (PyTorch/XLA)**: Default to a *lazy execution* model. Operations build a computation graph, which is then compiled by XLA and executed. For more information about lazy tensors, read about how [XLA Tensors are Lazy](./pytorch-on-xla-devices.md#xla-tensors-are-lazy).
This compilation step occurs when a graph is first encountered or if the graph structure/input shapes change. Subsequent runs with the same graph are much faster.

**Device Abstraction:**

* **GPUs**: Accessed via `torch.device("cuda:0")` or similar.
* **TPUs**: Accessed via `xm.xla_device()`.

**Distributed Training:**

* While `torch.nn.parallel.DistributedDataParallel` (DDP) is supported with an XLA backend, PyTorch/XLA also offers its own multi-processing utilities. (`torch_xla.launch` and `pl.MpDeviceLoader`) and advanced paradigms like SPMD that are often more performant on TPUs.

**Mixed Precision:**

* TPUs have native support for `bfloat16`, which is often preferred over `float16` for stability and performance, and typically doesn't require loss scaling.  Only certain GPUs have `bfloat16` support.

## Core Code Modifications

Here are the essential changes you'll need to make:

### Device Handling

Replace CUDA device specifications with XLA device specifications.

**(OLD) GPU Code:**

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
data = data.to(device)
```

**(NEW) PyTorch/XLA Code:**

```python
import torch_xla.core.xla_model as xm

# Acquire the XLA device (e.g., a TPU core)
device = xm.xla_device()

# Move your model and data to the XLA device
model.to(device)
data = data.to(device)
```

### Understanding Lazy Execution and Synchronization

This is the most significant conceptual shift. Since PyTorch/XLA executes lazily, you need to tell it when to actually compile and run the graph.

* **`torch_xla.sync()`**: These functions act as barriers. They signal to PyTorch/XLA that a portion of the graph is complete and should be compiled and executed.
  * In typical training loops, you'll call `torch_xla.sync()` once per iteration, usually after `optimizer.step()`.
  * When using PyTorch/XLA's multi-processing for distributed training, `xm.optimizer_step(optimizer)` handles this synchronization implicitly.

**(OLD) GPU Code (Eager Execution):**

```python
# ...
loss.backward()
optimizer.step() # Operations execute here
print(loss.item()) # Value is available
```

**(NEW) PyTorch/XLA Code (Lazy Execution):**

```python
# ...
loss.backward()
optimizer.step() # Operations are added to a graph, not executed yet

# Explicitly tell XLA to compile and run the graph for this step
torch_xla.sync()

# To print a tensor's value, ensure the graph producing it has executed.
# Often, printing is done less frequently or via asynchronous callbacks.
# If printing immediately after sync, the value will be available:
print(loss.item())
# For better performance, consider printing/logging less frequently or moving
# the tensor to CPU first if needed outside the main loop:
print(loss.cpu().item()) # This will also trigger a sync if not done already
```

## Performance and Scalability Optimizations

### Data Loading

For efficient data input on TPUs, especially in distributed settings, replace standard PyTorch `DataLoader` with `torch_xla.distributed.parallel_loader.MpDeviceLoader`.

**(OLD) GPU Code:**

```python
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
for data, target in train_loader:
    data, target = data.to(gpu_device), target.to(gpu_device)
    # ... training step ...
```

**(NEW) PyTorch/XLA Code (Multi-Device/Multi-Process):**

```python
import torch_xla.distributed.parallel_loader as pl

# Inside your _mp_fn (multi-processing function, see below)
# train_loader is your standard PyTorch DataLoader
# device is xm.xla_device() for the current process
mp_train_loader = pl.MpDeviceLoader(train_loader, device)

for batch_idx, (data, target) in mp_train_loader:
    # Data is already on the correct XLA device slice
    # ... training step ...
```

`MpDeviceLoader` handles prefetching data to the device, overlapping data transfers with computation.

### Distributed Training

While `torch.nn.parallel.DistributedDataParallel` can be used with an XLA backend (see [DDP Guide](../perf/ddp.md)), PyTorch/XLA's `torch_xla.launch` utility provides a common way to spawn multiple Python processes for distributed training, where each process typically controls one XLA device or a set of devices. This is often used with `torch_xla.distributed.parallel_loader.MpDeviceLoader` and `torch_xla.core.xla_model.optimizer_step`.

Let's adapt our single-device MNIST example to run in a distributed fashion using `torch_xla.launch`.

**Single-Device MNIST Snippet (Recap):**

```python
device = xm.xla_device()
model = MNISTNet().to(device)
train_loader = torch.utils.data.DataLoader(...) # Standard DataLoader
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

for data, target in train_loader:
    optimizer.zero_grad()
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    torch_xla.sync()
```

**PyTorch/XLA Multi-Process MNIST using** **`torch_xla.launch`:**

First, we define a main training function that will be executed by each process. Let's call it `_mp_mnist_fn`*.*

```python
# mnist_xla_distributed.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# PyTorch/XLA specific imports

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl # For MpDeviceLoader
import torch_xla.runtime as xr # For runtime utilities like world_size

# Re-define MNISTNet here for completeness (same as before)

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# This function will be executed by each XLA process

def _mp_mnist_fn(index, args):
    # `index` is the global ordinal of the current process (0 to N-1)
    # `args` are the arguments passed from torch_xla.launch

    torch.manual_seed(args.seed) # Ensure consistent model initialization if needed

    # 1. Acquire the XLA device for THIS process.
    device = xm.xla_device()

    # 2. Create the model and move it to the process-specific XLA device
    model = MNISTNet().to(device)

    # 3. Create the optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    loss_fn = nn.NLLLoss()

    # 4. Wrap the PyTorch DataLoader with MpDeviceLoader
    #    This handles distributing data shards to each device.
    #    The underlying train_dataset is shared, MpDeviceLoader handles per-device dispatch.
    mp_train_loader = pl.MpDeviceLoader(args.train_loader, device)

    print(f"Process {index} (Global Ordinal {xr.global_ordinal()}): Starting training on {xm.xla_device_hw(device)}...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(mp_train_loader):
            # Data and target are already on the correct 'device' from MpDeviceLoader
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()

            # 5. Use xm.optimizer_step to handle gradient reduction and optimizer update
            #    This also includes the necessary synchronization (like torch_xla.sync()).
            xm.optimizer_step(optimizer)

            if batch_idx % args.log_interval == 0:
                # Print only on the master ordinal to avoid spamming logs
                if xm.is_master_ordinal(local=False): # global_master=False for per-host master
                    print(f'Process {index} - Train Epoch: {epoch} '
                          f'[{batch_idx * len(data) * xr.world_size()}/{len(args.train_loader.dataset)} '
                          f'({100. * batch_idx / len(args.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # It's good practice to wait for all processes at epoch boundaries if doing validation
        # xm.rendezvous('epoch_end') # Example, might not be strictly needed here

    if xm.is_master_ordinal(local=False):
        print(f"Process {index}: Training finished!")
        # Optionally save the model (master_only is True by default in xm.save)
        # xm.save(model.state_dict(), "mnist_xla_distributed.pt")

# Define training configurations
class Args:
    epochs = 2
    learning_rate = 0.01
    momentum = 0.5
    batch_size = 64 # This will be the per-device batch size
    seed = 42
    log_interval = 50 # Log every 50 batches

if __name__ == '__main__':

    args = Args()

    # Create the shared DataLoader (not wrapped by MpDeviceLoader here yet)
    train_dataset = datasets.MNIST(
        './data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    # Note: No DistributedSampler needed here if using MpDeviceLoader,
    # as MpDeviceLoader handles distributing distinct shards.
    args.train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True, # Shuffle should be True for training
        num_workers=1 # Can be > 0
    )

    # 6. Start distributed training on all available XLA devices with torch_xla.launch
    # Ensure PJRT_DEVICE is set in your environment (e.g., export PJRT_DEVICE=TPU)
    print("Launching distributed MNIST training...")
    torch_xla.launch(_mp_mnist_fn, args=(args,))
```

**Key changes from the single-device script for distributed training:**

* **Main Training Logic in** \_**`mp_mnist_fn`**: The core training loop is encapsulated in a function that `torch_xla.launch` will execute in multiple processes.
* **`torch_xla.launch(_mp_mnist_fn, args=(args,))`**: This is the entry point. It spawns `N` processes (where `N` is the number of available XLA devices/chips, e.g., 8 for a TPU v3-8) and runs `mp_mnist_fn` in each, passing the `index` (global ordinal) and `args`.
* **`device = xm.xla_device()`** **inside** \_**`mp_mnist_fn`**: Each process gets its unique XLA device.
* **`pl.MpDeviceLoader(args.train_loader, device)`**: This wraps your standard `DataLoader`. `MpDeviceLoader` ensures that each process (and its device) gets a unique shard of the data from `args.train_loader`. It also typically handles `xm.mark_step()` internally after a configurable number of batches.
* **`xm.optimizer_step(optimizer)`**: This is crucial for distributed training. It performs an all-reduce operation on the gradients across all devices, averages them, and then applies the optimizer step. It also includes the necessary synchronization, so a separate `torch_xla.sync()` is usually not needed when using `xm.optimizer_step()`.
* **Logging with** **`xm.is_master_ordinal(local=False)`**: In distributed training, you often want to log or save checkpoints only from one process (the global master) to avoid redundant output or race conditions.

This example illustrates how to adapt a single-device script for multi-process data parallelism using PyTorch/XLA's idiomatic tools.

For an example 4 TPU setup, you should see output like the following:

```shell
Launching distributed MNIST training...
Process 0 (Global Ordinal 0): Starting training on TPU...
Process 3 (Global Ordinal 3): Starting training on TPU...
Process 1 (Global Ordinal 1): Starting training on TPU...
Process 2 (Global Ordinal 2): Starting training on TPU...
Process 0 - Train Epoch: 1 [0/60000 (0%)]	Loss: 2.315475
Process 0 - Train Epoch: 1 [12800/60000 (5%)]	Loss: 1.661397
Process 0 - Train Epoch: 1 [25600/60000 (11%)]	Loss: 0.468286
...
Process 0 - Train Epoch: 1 [230400/60000 (96%)]	Loss: 0.191208
Process 0 - Train Epoch: 2 [0/60000 (0%)]	Loss: 0.315434
Process 0 - Train Epoch: 2 [12800/60000 (5%)]	Loss: 0.163842
Process 0 - Train Epoch: 2 [25600/60000 (11%)]	Loss: 0.042292
...
Process 0 - Train Epoch: 2 [230400/60000 (96%)]	Loss: 0.050310
Process 0: Training finished!

```

### Automatic Mixed Precision (AMP)

TPUs excel with `bfloat16` precision, which typically doesn't require loss scaling. PyTorch/XLA’s AMP extends PyTorch's AMP, and automatically casts to `float32` or `bfloat16` on TPU devices.

**PyTorch/XLA Code (BF16 AMP on TPU):**

```python
from torch_xla.amp import syncfree
import torch_xla.core.xla_model as xm

# Creates model and optimizer in default precision
model = Net().to('xla')
# Pytorch/XLA provides sync-free optimizers for improved performance
optimizer = syncfree.SGD(model.parameters(), ...)

for input, target in data:
    optimizer.zero_grad()

    # Enables autocasting for the forward pass
    with autocast(torch_xla.device()):
        output = model(input)
        loss = loss_fn(output, target)

    # Exits the context manager before backward()
    loss.backward()
    xm.optimizer_step.(optimizer)
```

For more information about autoscaling, see the [AMP Guide](../perf/amp.md).

## Other Important Considerations

* **Saving and Loading Models**:

  * **GPU**: `torch.save(model.state_dict(), "model.pt")`
  * **PyTorch/XLA**: It's best practice to save state dicts from the CPU.

```python
# Saving
xm.save(model.state_dict(), "model_xla.pt", master_only=True) # master_only recommended
Or for more complex scenarios, especially if moving to other environments:
if xm.is_master_ordinal():
   cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
   torch.save(cpu_state_dict, "model_cpu.pt")

# Loading
model.load_state_dict(torch.load("model_cpu.pt"))
model.to(device) # Then move to XLA device
```

    PyTorch/XLA provides `xm.save()` which handles some XLA specifics. For maximum portability (e.g., loading on a non-XLA system), explicitly moving tensors to CPU before saving is safest. For sharded models (FSDP/SPMD), refer to specific [Distributed Checkpointing](../perf/spmd_distributed_checkpoint.md) docs.


* **Debugging and Profiling**:

  * GPU tools like `nvprof` or Nsight Systems won't work directly for XLA device code.
  * PyTorch/XLA provides its own [profiling tools](./xla-profiling.md) and metrics (`torch_xla.debug.metrics`).


* **Understanding Recompilations**:

  * A new concept for GPU users is graph recompilation. If the computation graph or input tensor shapes change between `torch_xla.sync()` calls, XLA needs to recompile, which can be slow.
  * Strive for static shapes and computation graphs within your main training/evaluation loops. Padding inputs to fixed sizes is a common strategy.
  * See [Understanding and Reducing Recompilations](../perf/recompilation.md).


* **Unsupported Operations**:

  * While PyTorch/XLA has extensive operator coverage, some PyTorch operations might not have an XLA lowering. These will fall back to CPU, causing slowdowns. Check the metrics report for `aten::` ops (see [Troubleshooting Basics](./troubleshoot.md)).

## Example Workflow Transformation (Conceptual)

Here is pseudo code that highlights the conceptual differences between GPU, single device TPU, and multi device TPU training:

**Typical PyTorch/GPU Script Structure:**

```python
# 1. Imports
# 2. Model, Optimizer, DataLoader, Loss_fn definitions
# 3. Move model to GPU
# 4. Training loop:
#    a. Move data to GPU
#    b. optimizer.zero_grad()
#    c. Forward pass
#    d. Loss calculation
#    e. loss.backward()
#    f. optimizer.step()
#    g. Logging/Metrics
```

**PyTorch/XLA on TPU Script Structure (Single Device):**

```python
# 1. Imports (include torch_xla, torch_xla.core.xla_model as xm)
# 2. Model, Optimizer, DataLoader, Loss_fn definitions
# 3. device = xm.xla_device()
# 4. Move model to XLA device
# 5. Training loop:
#    a. Move data to XLA device
#    b. optimizer.zero_grad()
#    c. Forward pass
#    d. Loss calculation
#    e. loss.backward()
#    f. optimizer.step()
#    g. torch_xla.sync() # Crucial step
#    h. Logging/Metrics (be mindful of .item() or .cpu() calls)
```

**PyTorch/XLA on TPU Script Structure (Multi-Process with `torch_xla.launch`):**

```python
# def _mp_fn(index, args):
#    # 1. device = xm.xla_device()
#    # 2. Model, Optimizer, Loss_fn definitions
#    # 3. Move model to XLA device
#    # 4. mp_loader = pl.MpDeviceLoader(args.dataloader, device)
#    # 5. Training loop:
#    #    a. Data from mp_loader is already on device
#    #    b. optimizer.zero_grad()
#    #    c. Forward pass
#    #    d. Loss calculation
#    #    e. loss.backward()
#    #    f. xm.optimizer_step(optimizer) # Handles sync and gradient reduction
#    #    g. Logging/Metrics (check master_ordinal for single print)

# if __name__ == '__main__':
#    # Setup Dataloader, etc. in args
#    torch_xla.launch(_mp_fn, args=(args,))
```
