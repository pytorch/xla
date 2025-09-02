import torch
import os
from typing import Any, Dict
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np


def _to_jax(pytree):
  return jax.tree_util.tree_map(
      lambda x: jnp.asarray(x.cpu().numpy())
      if isinstance(x, torch.Tensor) else x, pytree)


def _to_torch(pytree):
  return jax.tree_util.tree_map(
      lambda x: torch.from_numpy(np.asarray(x))
      if isinstance(x, (jnp.ndarray, jax.Array)) else x, pytree)


def save_checkpoint(state: Dict[str, Any], path: str, step: int):
  """Saves a checkpoint to a file in JAX style.

  Args:
    state: A dictionary containing the state to save. torch.Tensors will be
      converted to jax.Array.
    path: The path to save the checkpoint to. This is a directory.
    step: The training step.
  """
  state = _to_jax(state)
  checkpoints.save_checkpoint(path, state, step=step, overwrite=True)


def load_checkpoint(path: str) -> Dict[str, Any]:
  """Loads a checkpoint and returns it in JAX format.

  This function can load both PyTorch-style (single file) and JAX-style
  (directory) checkpoints.

  If the checkpoint is in PyTorch format, it will be converted to JAX format.

  Args:
    path: The path to the checkpoint.

  Returns:
    The loaded state in JAX format (pytree with jax.Array leaves).
  """
  if os.path.isdir(path):
    # JAX-style checkpoint
    state = checkpoints.restore_checkpoint(path, target=None)
    if state is None:
      raise FileNotFoundError(f"No checkpoint found at {path}")
    return state
  elif os.path.isfile(path):
    # PyTorch-style checkpoint
    state = torch.load(path, weights_only=False)
    return _to_jax(state)
  else:
    raise FileNotFoundError(f"No such file or directory: {path}")
