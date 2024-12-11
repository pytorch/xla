# Distributed Checkpointing

PyTorch/XLA SPMD is compatible with the
[torch.distributed.checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)
library through a dedicated `Planner` instance. Users are able to
synchronously save and load checkpoints through this common interface.

The SPMDSavePlanner and SPMDLoadPlanner
([src](https://github.com/pytorch/xla/blob/master/torch_xla/experimental/distributed_checkpoint.py))
classes enable the `save` and `load` functions to operate directly on
the shards of an `XLAShardedTensor`, enabling all of the benefits of
distributed checkpointing in SPMD training.

Here is a demonstration of the synchronous distributed checkpointing
API:

``` python
import torch.distributed.checkpoint as dist_cp
import torch_xla.experimental.distributed_checkpoint as xc

# Saving a state_dict
state_dict = {
    "model": model.state_dict(),
    "optim": optim.state_dict(),
}

dist_cp.save(
    state_dict=state_dict,
    storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
    planner=xc.SPMDSavePlanner(),
)
...

# Loading the model's state_dict from the checkpoint. The model should
# already be on the XLA device and have the desired sharding applied.
state_dict = {
    "model": model.state_dict(),
}

dist_cp.load(
    state_dict=state_dict,
    storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
    planner=xc.SPMDLoadPlanner(),
)
model.load_state_dict(state_dict["model"])
```

## CheckpointManager

The experimental
[CheckpointManager](https://github.com/pytorch/xla/blob/master/torch_xla/experimental/distributed_checkpoint/manager.py#L40)
interface provides a higher-level API over the
`torch.distributed.checkpoint` functions to enable a few key features:

-   **Managed checkpoints**: Each checkpoint taken by the
    `CheckpointManager` is identified by the step at which it was taken.
    All steps tracked are accessible through the
    `CheckpointManager.all_steps` method, and any tracked steps can be
    restored using `CheckpointManager.restore`.
-   **Asynchronous checkpointing**: Checkpoints taken through the
    `CheckpointManager.save_async` API are written to persistent storage
    asynchronously to unblock training for the duration of the
    checkpoint. The input sharded state_dict is first moved to CPU
    before the checkpoint is dispatched to a background thread.
-   **Auto-checkpointing on preemption**: On Cloud TPU, preemptions can
    be detected and a checkpoint taken before the process is terminated.
    To use, ensure your TPU is provisioned through a QueuedResource with
    [Autocheckpointing
    enabled](https://cloud.google.com/sdk/gcloud/reference/alpha/compute/tpus/queued-resources/create#--autocheckpoint-enabled),
    and ensure the `chkpt_on_preemption` parameter is set when
    constructing the CheckpointManager (this option is enabled by
    default).
-   **FSSpec Support**: `CheckpointManager` uses an fsspec storage
    backend to enable checkpointing directly to any fsspec-compatible
    filesystem, including GCS.

Example usage of the CheckpointManager is below:

``` python
from torch_xla.experimental.distributed_checkpoint import CheckpointManager, prime_optimizer

# Create a CheckpointManager to checkpoint every 10 steps into GCS.
chkpt_mgr = CheckpointManager('gs://my-bucket/my-experiment', 10)

# Select a checkpoint to restore from, and restore if applicable
tracked_steps = chkpt_mgr.all_steps()
if tracked_steps:
    # Choose the highest step
    best_step = max(tracked_steps)
    # Before restoring the checkpoint, the optimizer state must be primed
    # to allow state to be loaded into it.
    prime_optimizer(optim)
    state_dict = {'model': model.state_dict(), 'optim': optim.state_dict()}
    chkpt_mgr.restore(best_step, state_dict)
    model.load_state_dict(state_dict['model'])
    optim.load_state_dict(state_dict['optim'])

# Call `save` or `save_async` every step within the train loop. These methods
# return True when a checkpoint is taken.
for step, data in enumerate(dataloader):
    ...
    state_dict = {'model': model.state_dict(), 'optim': optim.state_dict()}
    if chkpt_mgr.save_async(step, state_dict):
        print(f'Checkpoint taken at step {step}')
```

### Restoring Optimizer State

In distributed checkpointing, the state_dicts are loaded in-place, and
only the required shards of the checkpoint are loaded. Since optimizer
states are lazily created, the state isn't present until the first
`optimizer.step` call, and attempts to load an unprimed optimizer will
fail.

The utility method `prime_optimizer` is provided for this: it runs a
fake train step by setting all gradients to zero and calling
`optimizer.step`. *This is a destructive method and will touch both
model parameters and optimizer state*, so it should only be called just
prior to restoration.

#### Process Groups

To use `torch.distributed` APIs such as distributed checkpointing, a
process group is required. In SPMD mode, the `xla` backend is not
supported since the compiler is responsible for all collectives.

Instead, a CPU process group such as `gloo` must be used. On TPUs, the
`xla://` init_method is still supported to discover the master IP,
global world size, and host rank. An example initialization is below:

``` python
import torch.distributed as dist
# Import to register the `xla://` init_method
import torch_xla.distributed.xla_backend
import torch_xla.runtime as xr

xr.use_spmd()

# The `xla://` init_method will automatically discover master worker IP, rank,
# and global world size without requiring environment configuration on TPUs.
dist.init_process_group('gloo', init_method='xla://')
```
