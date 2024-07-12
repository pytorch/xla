import fsspec
import logging
import os
import pickle
import threading
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.experimental.distributed_checkpoint as xc
import traceback

from dataclasses import dataclass
from datetime import datetime
from collections import deque
from fsspec.core import url_to_fs
from os.path import basename
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Deque, List, Optional, Union
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from ._helpers import _sharded_cpu_state_dict

# TODO(jonbolin): Import path will change
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter

# File to track manager-specific metadata within each checkpoint path
_MANAGER_METADATA_FILE = '.manager_metadata'


@dataclass
class _CheckpointMetadata:
  # The step at which the checkpoint was taken
  step: int

  # The time at which the checkpoint was taken
  ts: datetime


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
  >>>   # Before restoring the checkpoint, the optimizer state must be primed
  >>>   # to allow state to be loaded into it.
  >>>   prime_optimizer(optim)
  >>>   state_dict = {'model': model.state_dict(), 'optim': optim.state_dict()}
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
  https://github.com/google/orbax/blob/efc079c/checkpoint/orbax/checkpoint/checkpoint_manager.py
  """

  # The base path to write checkpoints to. Each checkpoint taken by the manager
  # will be written into a subdirectory of this path, identified by the
  # checkpoint's step.
  base_path: Union[str, os.PathLike]

  # The interval to take checkpoints, in steps.
  save_interval: int

  # The maximum number of checkpoints to keep.
  max_to_keep: int

  # Whether a checkpoint should be taken when a preemption is detected.
  chkpt_on_preemption: bool

  def __init__(self,
               path: str,
               save_interval: int,
               max_to_keep: Optional[int] = 0,
               max_pending_async: Optional[int] = 1,
               process_group: dist.ProcessGroup = None,
               chkpt_on_preemption: bool = True):
    """
    Create a checkpoint manager that reads and writes checkpoints into
    the provided directory.

    Args:
      path: The base path for the CheckpointManager to write checkpoints into.
      save_interval: The number of steps between saving checkpoints.
      max_to_keep: The maximum number of checkpoints to be tracked by the
            CheckpointManager. When a new checkpoint will be taken, the
            checkpoint for the lowest tracked step will be deleted.
            Default: 0, indicating no upper bound on the number of checkpoints.
      max_pending_async: The maximum number of async checkpoints which can be
            pending. This should be a small value to ensure training doesn't
            get too far ahead of the last finished checkpoint, but increasing
            the value can unblock training when there are transient issues which
            slow down the active checkpoint.
            Default: 1, which only allows a single async checkpoint to be
            pending at a time.
      process_group: The process group to use when coordinating the checkpoint.
            Default: None, in which case a subgroup of the default process
            group will be created.
      chkpt_on_preemption: Whether or not to take a checkpoint when a
            preemption has been detected.
            Default: True
    """
    assert dist.is_initialized(), "A process group is required."
    assert save_interval > 0, "save_interval must be positive"
    assert max_pending_async > 0, "max_pending_async must be positive"
    assert max_to_keep >= 0, "max_to_keep must be non-negative"

    self.base_path = os.path.join(path, '')  # Ensure the base path ends in '/'
    self.save_interval = save_interval
    self.max_to_keep = max_to_keep
    self.chkpt_on_preemption = chkpt_on_preemption

    # Create a new group if none is provided
    # TODO(jonbolin): Verify subgroup on GPU backend
    self.pg = process_group or dist.new_group()

    # Thread pool to run the async checkpoints. `_async_sem` is used to guard
    # the number of pending checkpoints, and `_async_futures` tracks all
    # futures returned by the pool.
    self._async_worker_pool = ThreadPoolExecutor(max_workers=1)
    self._async_sem = threading.Semaphore(max_pending_async)
    self._async_futures = []
    # Mutex to ensure only a single thread can write a checkpoint at a time.
    self._save_mutex = threading.Lock()

    self._tracked_chkpts = self._load_tracked_chkpts()

    if self.chkpt_on_preemption:
      # Initialize the distributed runtime for preemption detection
      torch_xla._XLAC._ensure_xla_coordinator_initialized(
          xr.process_index(), xr.process_count(), xr.get_master_ip())
      torch_xla._XLAC._activate_preemption_sync_manager()

  def _load_tracked_chkpts(self) -> Deque[_CheckpointMetadata]:
    """
    Loads a list of all tracked checkpoints from the storage backend.
    """
    all_chkpts = []
    invalid_paths = []
    fs, raw_path = url_to_fs(self.base_path)
    if not fs.exists(raw_path):
      fs.mkdir(raw_path)
    else:
      for path in fs.ls(raw_path, detail=False):
        try:
          with fs.open(os.path.join(path, _MANAGER_METADATA_FILE), 'rb') as f:
            all_chkpts.append(pickle.load(f))
        except:
          invalid_paths.append(path)

    if invalid_paths:
      logging.warning(f'Ignoring invalid checkpoints: {invalid_paths}')
    return deque(sorted(all_chkpts, key=lambda m: m.ts))

  def _get_path(self, step: int) -> str:
    return os.path.join(self.base_path, str(step))

  def _delete_chkpt_at_step(self, step):
    if dist.get_rank(self.pg) == 0:
      path = self._get_path(step)
      fs, raw_path = url_to_fs(path)
      if fs.exists(raw_path):
        fs.rm(raw_path, recursive=True)

  def _release_oldest_checkpoints(self):
    """
    Delete oldest checkpoints until the number of tracked checkpoints is below
    self.max_to_keep. This operation is only execution on the rank 0 process.
    """
    if self.max_to_keep > 0:
      while len(self._tracked_chkpts) > self.max_to_keep:
        oldest_chkpt = self._tracked_chkpts.popleft()
        self._delete_chkpt_at_step(oldest_chkpt.step)

  def _wait_for_data(self):
    xm.mark_step()
    xm.wait_device_ops()

  def _save(self, step, state_dict):
    """
    The actual checkpointing logic, which is shared between async and
    synchronous checkpointing.

    The caller must ensure that data is accessible within the state_dict before
    calling, which can be achieved with `self._wait_for_data`.
    """
    with self._save_mutex:
      path = self._get_path(step)
      # Delete any existing checkpoint at the current step.
      self._delete_chkpt_at_step(step)
      dist_cp.save(
          state_dict=state_dict,
          storage_writer=FsspecWriter(
              path,
              per_thread_copy_ahead=0,
          ),
          planner=xc.SPMDSavePlanner(),
          process_group=self.pg,
      )
      metadata = _CheckpointMetadata(step=step, ts=datetime.now())
      self._tracked_chkpts.append(metadata)
      if dist.get_rank(self.pg) == 0:
        with fsspec.open(os.path.join(path, _MANAGER_METADATA_FILE), 'wb') as f:
          pickle.dump(metadata, f)
        self._release_oldest_checkpoints()

  def should_save(self, step: int) -> bool:
    """
    Returns true if a checkpoint should be saved for the current step. A
    checkpoint should be taken if any of the following conditions are met:
     - The step aligns with the CheckpointManager's save_interval.
     - The CheckpointManager was created with the `chkpt_on_preemption` option
       and a preemption has been detected.
    """
    preemption_detected = False
    if self.chkpt_on_preemption and self.reached_preemption(step):
      logging.warning(
          f"Preemption sync point reached at step {step}. Triggering a checkpoint."
      )
      preemption_detected = True
    return step % self.save_interval == 0 or preemption_detected

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
    if self.should_save(step) or force:
      self._wait_for_data()
      self._save(step, state_dict)
      return True
    return False

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
    if self.should_save(step) or force:
      self._wait_for_data()
      # Move the state_dict to CPU
      cpu_state_dict = _sharded_cpu_state_dict(state_dict)
      self._async_sem.acquire()
      future = self._async_worker_pool.submit(self._save, step, cpu_state_dict)
      future.add_done_callback(lambda _: self._async_sem.release())
      self._async_futures.append(future)
      return True
    return False

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
    tracked_steps = set(x.step for x in self._tracked_chkpts)
    assert step in tracked_steps, f'Cannot restore from untracked step {step}. Valid steps are: {tracked_steps}'
    path = self._get_path(step)
    dist_cp.load(
        state_dict=state_dict,
        storage_reader=FsspecReader(path),
        planner=xc.SPMDLoadPlanner(),
        process_group=self.pg,
    )

  def all_steps(self) -> List[int]:
    """
    List all steps tracked by the CheckpointManager.
    """
    return sorted(x.step for x in self._tracked_chkpts)

  def join(self):
    """ Wait for any pending async checkpoints to complete. """
    wait(self._async_futures)

  def reached_preemption(self, step: int) -> bool:
    """ Returns True if a preemption has been detected at the given step. """
    assert self.chkpt_on_preemption, (
        "Preemption detection not enabled. Please set `chkpt_on_preemption` "
        " when creating the CheckpointManager")
    return torch_xla._XLAC._sync_point_reached(step)
