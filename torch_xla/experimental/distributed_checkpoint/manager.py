import torch.distributed.checkpoint as dist_cp
import torch_xla.experimental.distributed_checkpoint as xc

from typing import List, Optional
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE


class CheckpointManager:
  """
  The CheckpointManager class provides a higher-level wrapper around the
  torch.distributed.checkpoint APIs to manage checkpointing. It builds on top
  of those APIs to enable a few key features:
    - Per-step checkpointing: Each checkpoint taken by the CheckpointManager is
          identified by the step at which it was taken, and any step tracked
          by the CheckpointManager can be restored.
    - Async checkpointing: The torch.distributed.checkpoint APIs are
          synchronous, which will block training for the duration of the
          checkpoint. The CheckpointManager's save_async method can be used to
          offload checkpointing to a background thread, unblocking training
          while the checkpoint is written to persistent storage.
    - Automatic checkpointing: If the training process would be shut down due
          to a SIGTERM, the CheckpointManager will automatically take a
          checkpoint at the next step.
    - Native fsspec integration: Any storage protocol compatible with fsspec
          can be used with CheckpointManager.
  
  The intended usage of CheckpointManager is as follows:

  >>> # Create a CheckpointManager to checkpoint every 10 steps into GCS.
  >>> chkpt_mgr = CheckpointManager('gs://my-bucket/my-experiment', 10)
  
  >>> # Select a checkpoint to restore from, and restore if applicable
  >>> tracked_steps = chkpt_mgr.all_steps()
  >>> if tracked_steps:
  >>>   # Choose the highest step
  >>>   best_step = max(tracked_steps)
  >>>   state_dict = {'model': model.state_dict()}
  >>>   chkpt_mgr.restore(best_step, state_dict)
  >>>   model.load_state_dict(state_dict['model'])

  >>> # Call `save` or `save_async` every step within the train loop.
  >>> for step, data in enumerate(dataloader):
  >>>   ...
  >>>   state_dict = {'model': model.state_dict(), 'optim': optim.state_dict()}
  >>>   if chkpt_mgr.save_async(step, state_dict):
  >>>     print(f'Checkpoint taken at step {step}')

  By calling `save` or `save_async` every step, the CheckpointManager has the
  opportunity to take a checkpoint on steps which are out-of-cycle with its
  step_period, as would be the case in auto checkpointing.

  This class is inspired by Orbax's CheckpointManager, which can be found here:
  https://github.com/google/orbax/blob/efc079c4e5b437782a80138913d322cb3ed365c7/checkpoint/orbax/checkpoint/checkpoint_manager.py
  """

  def __init__(self,
               path: str,
               save_period: int,
               max_to_keep: Optional[int] = -1,
               async_queue_size: Optional[int] = 1):
    """
    Create a checkpoint manager that reads and writes checkpoints into
    the provided directory.

    Args:
      path: The base path for the CheckpointManager to write checkpoints into.
      save_period: The number of steps between saving checkpoints.
      max_to_keep: The maximum number of checkpoints to be tracked by the
            CheckpointManager. When a new checkpoint will be taken, the
            checkpoint for the lowest tracked step will be deleted.
            Default: -1, indicating no upper bound on the number of checkpoints.
      async_queue_size: The size of the execution queue which processes async
            checkpoints. This should be a small value to ensure training doesn't
            get too far ahead of the last finished checkpoint, but increasing
            the value to 2 can unblock training when there are transient
            network issues which slow down the active checkpoint.
            Default: 1, which only allows a single async checkpoint to be
            pending at a time.
    """
    raise NotImplementedError

  def should_save(self, step: int) -> bool:
    """
    Returns true if a checkpoint should be saved for the current step or if
    a preemption has been detected.
    """
    raise NotImplementedError

  def save(self,
           step,
           state_dict: STATE_DICT_TYPE,
           force: Optional[bool] = False) -> bool:
    """
    Take a checkpoint synchronously if `self.should_save(step)`.

    Args:
      step: The current training step.
      state_dict: The state dict to be checkpointed.
      force: Option to force a checkpoint to be taken regardless of the result
             of `should_save(step)`.
    Returns:
      True if a checkpoint was taken and False otherwise.
    """
    raise NotImplementedError

  def save_async(self,
                 step: int,
                 state_dict: STATE_DICT_TYPE,
                 force: Optional[bool] = False) -> bool:
    """
    Take a checkpoint asynchronously if `self.should_save(step)`. The
    input state_dict will be transferred to the CPU device using the
    `sharded_cpu_state_dict` function.

    This function will do the following:
    1. Transfer `state_dict` to the CPU device.
    2. Dispatch the checkpoint workload to an asynchronous execution 
       queue. This will block training until the ongoing async 
       checkpoint finishes when the queue is full.

    Args:
      step: The current training step.
      state_dict: The state dict to be checkpointed.
      force: Option to force a checkpoint to be taken regardless of the result
             of `should_save(step)`.
    Returns:
      True if a checkpoint was taken and False otherwise.
    """
    raise NotImplementedError

  def restore(self, step: int, state_dict: STATE_DICT_TYPE) -> None:
    """
    Restores the checkpoint taken at the given step into the state_dict. The
    caller is responsible for calling `model.load_state_dict` to restore any
    non-tensor values.

    Args:
      step: The step whose checkpoint is to be restored.
      state_dict: The state dict to restore the checkpoint into. Values are
                  updated in-place within the state_dict.
    """
    raise NotImplementedError

  def all_steps(self) -> List[int]:
    """
    List all steps tracked by the CheckpointManager.
    """
    raise NotImplementedError
